import math
import numpy as np
from numpy.random import SeedSequence, Generator
from opacus.data_loader import DPDataLoader
from opacus.grad_sample.grad_sample_module import GradSampleModule
from pathlib import Path
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from skellam_utils import SkellamMechanismPyTorch as SkellamMechanism
from skellam_utils import params_to_vec, set_grad_to_vec, clip_gradient

from flwr.common import ndarrays_to_parameters
from models import LinearClassificationNet, MNIST_CNN, ResNeXt

class CustomDataset(Dataset):
    def __init__(self, x,y, data_transforms=None):
        self.x = x
        self.y = y
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.data_transforms:
            return self.data_transforms(self.x[idx]), self.y[idx]
        else:
            return self.x[idx], self.y[idx]

def get_initial_model_params(client):
    return ndarrays_to_parameters(client.get_parameters(None))

def get_model(model_name, data_dims):
    if model_name == 'CIFAR10_linear_classifier':
        return LinearClassificationNet(num_inp=data_dims, num_out=10, bias=True)
    elif model_name == 'MNIST_CNN':
        return MNIST_CNN()
    elif model_name == 'ResNext29':
        return ResNeXt(input_shapes=data_dims, num_classes=10, pretrained=True)
    elif model_name == 'Income_classifier':
        return LinearClassificationNet(num_inp=data_dims, num_out=2, bias=True)
    else:
        raise NotImplementedError(f'Model {model_name} not implemented!')

def get_dataloader(path_to_data:Path, train_set:bool, silo_testset_rng:Generator, general_rng:Generator, testset_frac:float, validation_frac:float, eval_dataset:str, client_number:int, batch_size:int, local_sampling_frac:float, workers:int=0, dataset: dict=None, shuffle:bool=True, drop_last:bool=False) -> DataLoader:

    pytorch_generator = torch.Generator(device='cpu')
    pytorch_seed = int(general_rng.integers(0, 2**32-1))
    pytorch_generator.manual_seed(pytorch_seed)

    if path_to_data is not None and path_to_data != '':
        tmp = f"{path_to_data}/client_{client_number}.npz"
        dataset = np.load(tmp, allow_pickle=True)

    x = np.array(dataset['x'], dtype=np.float32)
    y = np.array(dataset['y'], dtype=np.int64)

    # cross-silo train-val-test split
    if silo_testset_rng is not None:
        if (testset_frac is not None and testset_frac > 0) or (validation_frac is not None and validation_frac > 0):

            test_indices = silo_testset_rng.choice(np.arange(0, len(y)), int(len(y)*testset_frac), replace=False)
            if eval_dataset == 'val':
                x = np.delete(x, test_indices, axis=0)
                y = np.delete(y, test_indices, axis=0)
                val_indices = general_rng.choice(np.arange(0, len(y)), int(len(y)*validation_frac), replace=False)
                if not train_set:
                    x = x[val_indices]
                    y = y[val_indices]
                else:
                    x = np.delete(x, val_indices, axis=0)
                    y = np.delete(y, val_indices, axis=0)
            elif eval_dataset == 'test':
                if not train_set:
                    x = x[test_indices]
                    y = y[test_indices]
                else:
                    x = np.delete(x, test_indices, axis=0)
                    y = np.delete(y, test_indices, axis=0)
            else:
                raise ValueError(f'Unknown eval dataset option: {eval_dataset}!')

    dataset = CustomDataset(x=torch.from_numpy(x), y=torch.from_numpy(np.array(y)))

    if batch_size is None:
        batch_size = int(np.fix(local_sampling_frac*len(dataset)))
        assert batch_size > 0, f'Batch size is {batch_size}!'
    else:
        batch_size = int(batch_size)

    kwargs = {"num_workers": workers, "pin_memory": True, "shuffle": shuffle, "drop_last": drop_last, "generator": pytorch_generator}    
    return DataLoader(dataset, batch_size=batch_size, **kwargs)

def get_seeds(n_rng, initial_seed=None):
    ss = SeedSequence(initial_seed)
    tmp = ss.spawn(n_rng)
    return np.array([seq.generate_state(1) for seq in tmp]).reshape(-1)

def set_seeds(seed:int):
    seed = int(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_random_states():
    tmp = {"numpy": np.random.get_state(), "random": random.getstate(), "torch": torch.get_rng_state(), }
    if torch.cuda.is_available():
        tmp["torch_cuda_all": torch.cuda.get_rng_state_all()]
    return tmp

def set_random_states(all_random_states):
    np.random.set_state(all_random_states["numpy"])
    random.setstate(all_random_states["random"])
    torch.set_rng_state(all_random_states["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(all_random_states["torch_cuda_all"])

def set_model_params(net, vec):
    for param in net.parameters():
        if not param.requires_grad:
            continue
        size = param.data.view(1, -1).size(1)
        param.data = vec[:size].view_as(param.data).clone()
        vec = vec[size:]

def flower_train(net, trainloader, epochs, optimiser_params, device:str, l2_clip:float, gaussian_noise_sigma:float, skellam_noise_sigma:float, max_physical_batchsize:int, run_only_batches:bool, use_skellam:str='all', quantization:int=32, skellam_num_clients:int=10):
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimiser_params.pop('optimizer')
    optimizer = torch.optim.SGD(net.parameters(), **optimiser_params)
    batch_size = trainloader.batch_size # expected batch size when using Poisson sampling
    trainloader = DPDataLoader.from_data_loader(trainloader)
    net.train()
    correct, loss_tot = 0, 0.
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    net = GradSampleModule(net, loss_reduction="sum")
    
    if use_skellam is not None:
        mu = (skellam_noise_sigma * l2_clip)**2
        skellam = SkellamMechanism(budget=quantization, d=num_params, norm_bound=l2_clip, mu=mu, device=device, num_clients=skellam_num_clients)
    
    if run_only_batches:
        epochs_to_run = 1
    else:
        epochs_to_run = epochs

    for i_epoch in range(epochs_to_run):
        for i_batch, (data, target) in enumerate(trainloader):
            if run_only_batches:
                if i_batch == epochs:
                    break
            loss_sum, acc_sum = 0., 0.
            grad_sum = torch.zeros(num_params).to(device)
            num_microbatches = int(math.ceil(float(data.size(0)) / max_physical_batchsize))
            final_microbatch = False
            same_rotation_batch = True

            for i_micro in range(num_microbatches):
                if i_micro == num_microbatches - 1:
                    final_microbatch = True
                for p in net.parameters():
                    if p.requires_grad:
                        p.grad = None
                        p.grad_sample = None
                        p.summed_grad = None

                lower = i_micro * max_physical_batchsize
                upper = min((i_micro+1) * max_physical_batchsize, data.size(0))
                x, y = data[lower:upper], target[lower:upper]
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                predicted = torch.argmax(outputs.detach(), 1)
                loss = criterion(outputs, y.to(torch.int64))
                loss_sum += loss.detach().item()/len(target)
                acc_sum += torch.sum(predicted == y).item()
                loss.backward()
                grads = params_to_vec(net, return_type="grad_sample")
                if len(grads.shape) == 1:
                    grads = grads.unsqueeze(0)

                if use_skellam is not None and use_skellam == 'all':
                    grads = clip_gradient(norm_clip=l2_clip, linf_clip=0., grad_vec=grads)
                    grads = skellam.add_noise(grad_vec=grads, same_rotation_batch=same_rotation_batch, final_microbatch=final_microbatch)
                else:
                    if l2_clip > 0:
                        grads = clip_gradient(norm_clip=l2_clip, linf_clip=0., grad_vec=grads)
                    if gaussian_noise_sigma > 0 and final_microbatch:
                        if len(grads.shape) == 1:
                            raise ValueError(f'Gradient vector shape is 1d, should be > 1d!')
                        grads[0] += l2_clip * torch.normal(mean=0., std=gaussian_noise_sigma, size=grads[0].shape).to(device)
                grad_sum += grads.sum(0)

            grad_mean = grad_sum / len(target)
            set_grad_to_vec(net, grad_mean)
            optimizer.step()
            correct += acc_sum
            loss_tot += loss_sum

    if run_only_batches:
        accuracy = correct / (epochs*batch_size)
        loss_tot = loss_tot / epochs
    else:
        accuracy = correct / (len(trainloader.dataset)*epochs)
        loss_tot = loss_tot / (len(trainloader)*epochs)
    return loss_tot, accuracy

def flower_test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    correct, loss, = 0, 0.0
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(testloader, disable=True):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss += criterion(outputs, y).item()/len(testloader.dataset)
            predicted = torch.argmax(outputs.data, 1)
            correct += torch.sum(predicted == y).item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def save_client_data(client_data, folder_name, data_props, N_is):
    print('Writing out client data..')
    for i_client, d in enumerate(client_data):
        # create folder if needed and save data, create subfolder for each client
        tmp = folder_name+'/'+str(i_client)
        Path(tmp).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tmp+f'/client_{i_client}', x = d['x'], y=d['y'], N_is=N_is[i_client], data_props=data_props[i_client], allow_pickle=True)
        print(i_client, ' done')

def do_client_split(x, y, num_clients):
    """Do iid split of data between clients
    """
    N = len(y)
    print('Doing iid split!')
    client_data = []
    data_props = []
    N_is = []
    all_labels = np.unique(y)

    # split each label into num_clients parts
    for i_label, label in enumerate(all_labels):
        inds = np.where(y==label)[0]
        split_inds = np.array_split(inds, num_clients)
        for i_client in range(num_clients):
            if i_label == 0:
                client_data.append({'x':[], 'y':[]})
            client_data[i_client]['x'].append(x[split_inds[i_client], :])
            client_data[i_client]['y'].append(y[split_inds[i_client]])
    data_props = []
    # shuffle client data & concat
    for i_client in range(num_clients):
        client_data[i_client]['x'] = np.concatenate(client_data[i_client]['x'], axis=0)
        client_data[i_client]['y'] = np.concatenate(client_data[i_client]['y'], axis=0)
        shuffle_inds = np.random.permutation(client_data[i_client]['x'].shape[0])
        client_data[i_client]['x'] = client_data[i_client]['x'][shuffle_inds, :]
        client_data[i_client]['y'] = client_data[i_client]['y'][shuffle_inds]            
        # calculate proportion of different labels on each client
        data_props.append(np.unique(client_data[i_client]['y'], return_counts=True)[1]/client_data[i_client]['y'].shape[0])
        # calculate client sizes
        N_is.append(client_data[i_client]['x'].shape[0])
    return client_data, N_is, data_props

def get_image_data(num_clients, dataset_name, data_folder, random_seed):
    
    # Load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # from pytorch basic examples
        testset = datasets.CIFAR10(root=data_folder, train=False, download=True, transform=transform)
        x, y = testset.data, testset.targets
        trainset = datasets.CIFAR10(root=data_folder, train=True, download=True, transform=transform)
        x2, y2 = trainset.data, trainset.targets
        x, y = np.concatenate((x, x2)), np.concatenate((y, y2))
        x = np.transpose(x, (0,3,1,2))
        del trainset, testset
        print('full data shapes for CIFAR10, x: {}, y: {}'.format(x.shape, y.shape))
    elif dataset_name == 'fashion_mnist':
        transform=transforms.Compose([
        transforms.ToTensor(),
        ])
        trainset = datasets.FashionMNIST(root=data_folder, train=True, download=True, transform=transform)
        x, y = trainset.data, trainset.targets
        testset = datasets.FashionMNIST(root=data_folder, train=False, download=True, transform=transform)
        x2, y2 = testset.data, testset.targets
        x, y = np.concatenate((x, x2)), np.concatenate((y, y2))
        del trainset, testset
        x = np.expand_dims(x, axis=1)
        print('full data shapes for Fashion MNIST, x: {}, y: {}'.format(x.shape, y.shape))
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}!')
    return do_client_split(x, y, num_clients)    
    
def get_feature_info(dataset_name: str):
    if dataset_name == 'cifar10-pretrained':
        data_dims = (1024)
        data_transforms = None
    elif dataset_name == 'fashion_mnist':
        data_dims = (1,28,28)
        data_transforms = None
    elif dataset_name == 'income':
        data_dims = (53)
        data_transforms = None
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}!")
    return data_dims, data_transforms