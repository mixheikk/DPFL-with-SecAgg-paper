
from jsonargparse import ArgumentParser, Namespace, ActionConfigFile
from collections import OrderedDict
import flwr as fl
from logging import DEBUG
import numpy as np
from pathlib import Path
import torch
from typing import Dict, Callable, Tuple, List, Dict
import time
import wandb

from flwr_strategies import FedAvg
from flwr.common.logger import log
from flwr.server.app import _init_defaults
from flwr.server.client_manager import SimpleRngClientManager
import flwr.simulation as simulation

from utils import get_seeds, set_seeds, get_model, get_dataloader, get_initial_model_params, get_feature_info
from utils import flower_test as test
from utils import flower_train as train

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid:str, fed_dir_data:str, eval_dataset:str, validation_frac:float, testset_frac:float, avail_clients:List, model_name:str, data_dims:Tuple, l2_clip:float, gaussian_noise_sigma:float, skellam_noise_sigma:float, max_physical_batchsize:int, use_skellam:str, quantization:int, skellam_num_clients:int, data_transforms=None, run_only_batches:bool=False, wait_on_eval:int=0, silo_testset_rng=None):
        self.cid = cid # id for flwr
        self.client_number = avail_clients[int(cid)] # flwr id mapped to train/validation client split
        self.fed_dir = Path(fed_dir_data + '/' +  str(self.client_number))
        self.eval_dataset = eval_dataset
        self.validation_frac = validation_frac
        self.testset_frac = testset_frac
        self.l2_clip = l2_clip
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.skellam_noise_sigma = skellam_noise_sigma
        self.max_physical_batchsize = max_physical_batchsize
        self.use_skellam = use_skellam
        self.quantization = quantization
        self.skellam_num_clients = skellam_num_clients
        self.data_transforms = data_transforms
        self.run_only_batches = run_only_batches # this affects only training, not evaluation
        self.use_skellam = use_skellam
        self.wait_on_eval = wait_on_eval
        self.silo_testset_rng = silo_testset_rng
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = get_model(model_name=model_name, data_dims=data_dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        
        set_params(self.net, parameters)
        general_rng = np.random.default_rng([self.client_number, config['FL_round_seed']])
        local_seed = general_rng.integers(0, 2**32-1)
        set_seeds(local_seed)
        self.net.to(self.device)
        optimiser_params = {
                            'optimizer' : config['optimizer'], 
                            'lr': config['lr_client'], 
                            'weight_decay' : config['weight_decay'], 
                            'momentum': config['client_momentum'],
                            }

        num_workers = 0 # can't use multiple workers with opacus
        trainloader = get_dataloader(
            path_to_data=self.fed_dir,
            client_number=self.client_number,
            batch_size=config["batch_size"],
            local_sampling_frac=config["local_sampling_frac"],
            workers=num_workers,
            train_set=True,
            dataset=None,
            silo_testset_rng = self.silo_testset_rng,
            general_rng = general_rng,
            eval_dataset=self.eval_dataset,
            validation_frac=self.validation_frac,
            testset_frac=self.testset_frac,
        )
        
        loss, acc = train(self.net, trainloader, epochs=config["epochs"], optimiser_params=optimiser_params, device=self.device,  l2_clip=self.l2_clip, gaussian_noise_sigma=self.gaussian_noise_sigma, skellam_noise_sigma=self.skellam_noise_sigma, max_physical_batchsize=self.max_physical_batchsize, run_only_batches=self.run_only_batches, use_skellam=self.use_skellam, quantization=self.quantization, skellam_num_clients=self.skellam_num_clients)
        n_losses = len(trainloader)
        return get_params(self.net), n_losses, {'accuracy': acc, 'loss': loss}

    def evaluate(self, parameters, config, num_workers = None):
        set_params(self.net, parameters)
        general_rng = np.random.default_rng([self.client_number, config['FL_round_seed']])
        local_seed = general_rng.integers(0, 2**32-1)
        set_seeds(local_seed)
        if num_workers is None:
            num_workers = 0 # can't use multiple workers with opacus 
        time.sleep(self.wait_on_eval)
        self.net.to(self.device)
        testloader = get_dataloader(
            path_to_data=self.fed_dir,
            client_number=self.client_number,
            batch_size=50,
            local_sampling_frac=config["local_sampling_frac"],
            workers=num_workers,
            train_set=False,
            dataset=None,
            silo_testset_rng = self.silo_testset_rng,
            general_rng = general_rng,
            eval_dataset=self.eval_dataset,
            validation_frac=self.validation_frac,
            testset_frac=self.testset_frac,
            )
        n_losses = len(testloader.dataset)
        loss, acc = test(self.net, testloader, device=self.device)
        return float(loss), n_losses, {'accuracy': acc, 'loss': loss}

    def get_client_model(self):
        return self.net

def get_config_fn(args:Namespace) -> Callable[[int], Dict]:
    """Return a (possibly) dynamic configuration."""
    def config_fn(server_round: int) -> Dict:
        config = {
            "FL_round_seed" : [server_round, args.init_seed], 
            "testset_frac": args.testset_frac,
            "validation_frac" : args.validation_frac,
            "eval_dataset" : args.eval_dataset,
            "epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "local_sampling_frac": args.local_sampling_frac,
            "optimizer" : args.client_optimizer,
            "lr_client": args.client_lr,
            "weight_decay" : args.client_weight_decay,
            "client_momentum" : args.client_momentum,
            "num_clients": args.num_clients,
        }
        return config
    return lambda server_round: config_fn(server_round=server_round)

def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)

def get_fit_aggregation_fn():

    def fit_aggregate_metrics(metrics_list):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics_list]
        losses = [num_examples * m["loss"] for num_examples, m in metrics_list]
        unweighted_losses = [m["loss"] for _, m in metrics_list]
        samples = [num_examples for num_examples, _ in metrics_list]
        ordered_acc = sorted([m["accuracy"] for _, m in metrics_list])
        min_client_acc = np.min(ordered_acc)
        max_client_acc = np.max(ordered_acc)
        to_return = {"client_weighted_accuracy": sum(accuracies) / sum(samples), 
                "client_weighted_loss": sum(losses) / sum(samples), 
                "client_mean_loss": np.mean(unweighted_losses),
                "client_mean_accuracy": np.mean(ordered_acc),
                "client_std_accuracy": np.std(ordered_acc),
                "min_client_accuracy": min_client_acc,
                "max_client_accuracy": max_client_acc,
        }
        return to_return
    
    return lambda metrics_list: fit_aggregate_metrics(metrics_list)

def get_eval_aggregation_fn():

    def eval_aggregate_metrics(metrics_list):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics_list]
        losses = [num_examples * m["loss"] for num_examples, m in metrics_list]
        unweighted_losses = [m["loss"] for _, m in metrics_list]
        samples = [num_examples for num_examples, _ in metrics_list]
        ordered_acc = sorted([m["accuracy"] for _, m in metrics_list])
        min_client_acc = np.min(ordered_acc)
        max_client_acc = np.max(ordered_acc)        
        to_return = {"client_weighted_accuracy": sum(accuracies) / sum(samples), 
                "client_weighted_loss": sum(losses) / sum(samples), 
                "client_mean_loss": np.mean(unweighted_losses),
                "client_mean_accuracy": np.mean(ordered_acc),
                "client_std_accuracy": np.std(ordered_acc),
                "min_client_accuracy": min_client_acc,
                "max_client_accuracy": max_client_acc,
        }

        log(DEBUG, f"Validation client weighted loss {sum(losses) / sum(samples)}, accuracy {sum(accuracies) / sum(samples)}")
        return to_return
    
    return lambda metrics_list: eval_aggregate_metrics(metrics_list)

def main(args):
    
    print('Starting main with data set name:', args.dataset_name, ', model name: ', args.model_name)
    
    if args.debug:
        init_seed = args.init_seed
        print(f'Debug run, using given initial seed: {init_seed}')
    else:
        init_seed = get_seeds(n_rng=1)[0]
        args.init_seed = init_seed
        print(f'Using initial seed: {init_seed}')

    if args.batch_size is not None and args.batch_size != "None" and int(args.batch_size) > 0:
        args.local_sampling_frac = None
        print(f'Using batch size: {args.batch_size}, ignoring local_sampling_frac')
    else:
        if args.batch_size is not None:
            args.batch_size = None
        args.local_sampling_frac = float(args.local_sampling_frac)
        print(f'Using local sampling fraction: {args.local_sampling_frac}')

    set_seeds(init_seed)
    training_clients = np.arange(args.num_clients)

    if args.l2_clip == 0:
        print('Running without DP (got l2_clip=0)!')
    else:
        assert args.l2_clip > 0, f"l2_clip value needs to be positive or 0, got {args.l2_clip}!"
    if args.use_skellam is not None:
        assert args.skellam_noise_sigma >= 0, f"Noise std needs to be non-negative, got skellam std {args.skellam_noise_sigma}!"
        if args.use_skellam == 'None':
            args.use_skellam = None
        else:
            assert args.use_skellam == 'all', f"use_skellam needs to be one of ['all', 'last'], got {args.use_skellam}!"
            assert args.quantization == 32, f"quantization should be 32 with skellam mechanism, got {args.quantization}!"
    else:
        assert args.gaussian_noise_sigma >= 0, f"Noise std needs to be non-negative, got Gaussian std {args.gaussian_noise_sigma}!"
    
    # refactor wandb tags
    wandb_tags = []
    if args.wandb_tags is not None and len(args.wandb_tags) > 0:
        tmp = args.wandb_tags.split(sep=',')
        for tag in tmp:
            wandb_tags.append(tag)

    # initialize wandb
    assert not (args.enable_wandb and args.do_wandb_sweep), "Can't do both wandb sweep and normal logging!"
    if args.enable_wandb:
        wandb.init(
                project=f"{args.wandb_project}",
                name=f"{args.wandb_run}",
                config=args,
                tags=wandb_tags,
            )
    elif args.do_wandb_sweep:
        wandb.init(
                project=f"{args.wandb_project}",
                config=args,
                tags=wandb_tags,
            )
        args.enable_wandb = True

    # setup client config fun
    fit_config = get_config_fn(args)
    eval_config = get_config_fn(args)

    client_resources = {
        "num_cpus": args.n_cpus_for_clients, 
        "num_gpus": args.frac_gpus_for_clients,
    }

    data_dims, data_transforms = get_feature_info(args.dataset_name)
    fed_dir = args.client_data_folder

    def client_fn(cid: str):
        return FlowerClient(cid=cid, fed_dir_data=fed_dir, eval_dataset=args.eval_dataset, validation_frac=args.validation_frac, testset_frac=args.testset_frac, avail_clients=training_clients, model_name=args.model_name, data_dims=data_dims, l2_clip=args.l2_clip, gaussian_noise_sigma=args.gaussian_noise_sigma, skellam_noise_sigma=args.skellam_noise_sigma,max_physical_batchsize=args.max_physical_batchsize, use_skellam=args.use_skellam, quantization=args.quantization, skellam_num_clients=args.skellam_num_clients, data_transforms=data_transforms, run_only_batches=args.run_only_batches, wait_on_eval=args.wait_on_eval, silo_testset_rng=np.random.default_rng([cid, args.train_test_split_seed]))

    # (optional) specify Ray config, might need to set these on cluster
    ray_num_cpus = args.n_cpus
    ray_num_gpus = args.n_gpus
    ram_memory = 16 * 1024* 1024 * 1024
    ray_init_args = {"include_dashboard": False, 
                    '_temp_dir':args.tmp_dir,
                    "num_cpus": ray_num_cpus,
                    "num_gpus": ray_num_gpus,
                    "_memory": ram_memory,
                    #"_redis_max_memory": 128 * 1024 * 1024,
                    #"object_store_memory": 128 * 1024 * 1024, 
                    }

    initial_parameters = get_initial_model_params(client_fn('0'))
        
    strategy = FedAvg(
        fraction_fit=args.client_sampling_frac,
        fraction_evaluate=args.per_round_eval_frac, 
        min_fit_clients=1.,
        min_evaluate_clients=0.,
        min_available_clients=len(training_clients),  # All clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        accept_failures=False, 
        fit_metrics_aggregation_fn=get_fit_aggregation_fn(),
        evaluate_metrics_aggregation_fn=get_eval_aggregation_fn(), 
        initial_parameters=initial_parameters,
        server_learning_rate=args.server_lr,
        server_momentum=args.server_momentum,
    )

    client_manager_seed = np.random.default_rng(args.init_seed).integers(0, 2**32-1)
    client_manager = SimpleRngClientManager(rng=np.random.default_rng(client_manager_seed))

    # init server
    initialized_config = fl.server.ServerConfig(num_rounds=args.global_rounds)
    initialized_server, _ = _init_defaults(
        server=None,
        config=initialized_config,
        strategy=strategy,
        client_manager=client_manager,
        random_seeds= {'init_seed': args.init_seed, 
                       'client_manager_seed': client_manager_seed, 
                       'testset_seed': args.train_test_split_seed
                       },
        enable_wandb=args.enable_wandb,
    )
    
    print('Starting simulation')
    hist = simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(training_clients),
        client_resources=client_resources,
        server=initialized_server,
        config=fl.server.ServerConfig(num_rounds=args.global_rounds),
        ray_init_args=ray_init_args,
    )

    wandb.finish()
    print('Simulation finished.\n')
    return hist

if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")
    parser.add_argument('--client_data_folder', default='data/income/federated/inherent-clients51-testclients0', type=str, help='Path to client data folder. Note: this should contain subfolders for each client.')
    parser.add_argument('--dataset_name', default='income', type=str, choices=['fashion_mnist','cifar10-pretrained','income'])
    parser.add_argument('--model_name', default='Income_classifier', type=str, choices=['MNIST_CNN', 'CIFAR10_linear_classifier','Income_classifier'])
    parser.add_argument('--num_clients', default=51, type=int, help="Total number of train/val clients. Note: doesn't include possible separate test set of clients.") # CHECK
    parser.add_argument('--client_sampling_frac', default=.1, type=float, help='Fraction of clients to sample for learning each global round.')
    parser.add_argument('--per_round_eval_frac', default=.1, type=float, help='Fraction of clients to use for evaluation on each FL round.')
    parser.add_argument("--eval_dataset", default='val', help="'test': use separate testset for evaluation, 'val': split training set into train-validation and evaluate on validation.", choices=['test', 'val'])
    parser.add_argument('--testset_frac', default=.2, type=float, help="Fraction of clients' data to use as testing set.")
    parser.add_argument('--validation_frac', default=.3, type=float, help="Fraction of clients' train data to use for validation, note: this only splits training data and is only used when eval_dataset='val', otherwise will use testset_frac.")
    parser.add_argument('--n_cpus', type=int, default=1, help='Number of cpus. To have more clients concurrently using shared gpus, each client should have at least 1 cpu.')
    parser.add_argument('--n_gpus', type=int, default=0, help='Number of gpus for ray (should match the actual number).')
    parser.add_argument('--n_cpus_for_clients', type=int, default=1, help='Number of cpus for each client.')
    parser.add_argument('--frac_gpus_for_clients', type=float, default=.0, help='Number of gpus for each client.')
    parser.add_argument('--train_test_split_seed', default=2303, type=int, help='Random seed used only for train-test split on each client. Keeping this fixed ensures that same internal train-test-split is done for each run.')
    parser.add_argument('--init_seed', default=2303, type=int, help='Initial random seed, used for everything else but original train-test splits on clients. NOTE: only used if debug==True, otherwise will generate random initial seed, which is then logged with wandb.')
    parser.add_argument('--server_lr', default=1., type=float, help='Learning rate for server.')
    parser.add_argument('--client_lr', default=.1, type=float, help='Learning rate for local optimisation.')
    parser.add_argument("--client_optimizer",type=str,default="sgd", help="Which optimizer to use for clients. Note: only SGD used.", choices=['sgd'])
    parser.add_argument('--client_weight_decay', default=0.0, type=float, help='Weight decay for clients.')
    parser.add_argument('--client_momentum', default=0.0, type=float, help='Momentum for clients.')
    parser.add_argument('--server_momentum', default=0.0, type=float, help='Momentum for server.')
    parser.add_argument('--batch_size', default=None, help="int, Minibatch size for local optimisation. Overrides the following local_sampling_frac if not None or 'None'.")
    parser.add_argument('--local_sampling_frac', default=.1, help='Float, minibatch sampling fraction for local optimisation. Overridden by batch_size if it is not None.')
    parser.add_argument('--local_epochs', default=1, type=int, help='Number of local epochs (or batches when run_only_batches is True) to optimize per global round.')
    parser.add_argument('--run_only_batches', action='store_true', default=False, help="If True will run local_epochs number of local STEPS instead of local EPOCHS.")
    parser.add_argument('--global_rounds', default=1, type=int, help='Number of global FL rounds to run.')
    #
    # dp settings
    parser.add_argument('--l2_clip', default=1., type=float, help='Max l2-norm bound for clipping. Note: for non-DP, use big clipping with 0 noise.')
    parser.add_argument('--gaussian_noise_sigma', default=0.0, type=float, help='Gaussian mechanism noise std when using Gaussian noise.')
    parser.add_argument('--skellam_noise_sigma', default=.5, type=float, help='Skellam mechanism noise std when using Skellam noise.')
    parser.add_argument('--use_skellam', default='all', type=str, help='Whether to use Skellam mechanism for DP. Should be "all" for Skellam DP noise. If None, will use Gaussian mechanism.')
    parser.add_argument('--quantization', default=32, type=int, help='Number of bits for quantization when using Skellam mechanism. Should be 32, ie, no quantization.')
    parser.add_argument('--skellam_num_clients', default=51, type=int, help='Number of clients for calculating scaliong in Skellam mechanism. This should match the number of training clients in FL round.')
    parser.add_argument('--max_physical_batchsize', default=1024, type=int, help='Max physical batchsize.')
    #
    # wandb initilization
    parser.add_argument("--enable-wandb", action='store_true', default=False, help='Use Wandb to log a regular train run.')
    parser.add_argument("--do-wandb-sweep", action='store_true', default=False, help='Run Wandb hyperparam sweep. Note: if True, then should have enable-wandb=False (results are still logged with wandb).')
    parser.add_argument("--wandb-project",type=str,default="local-steps-with-secsum")
    parser.add_argument("--wandb-run",type=str,default="Local steps with SecSum testing")
    parser.add_argument("--wandb-tags",type=str, default=None, help="Str of tags to attach for wandb separated by ','. Use empty string for no tags.")
    #
    # some extra args
    parser.add_argument('--tmp_dir', default=None, type=str, help='Ray tmp folder')
    parser.add_argument('--wait_on_eval', default=0, type=int, help='Wait for given number of secs before running eval. This is just to try and allow Ray to kill previous training clients and release GPUs if nothing else helps.')
    parser.add_argument("--config", action=ActionConfigFile, help="Separate config file name. This will overwrite corresponding defaults if present.")
    parser.add_argument("--debug", action='store_true', default=True, help="If True, use given initial PRNG seed for model training, otherwise generate a random initial seed")
    args = parser.parse_args()
    
    main(args)