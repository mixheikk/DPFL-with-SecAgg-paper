

from jsonargparse import ArgumentParser
from folktables import ACSDataSource, ACSIncome
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from utils import get_model, get_image_data, save_client_data, get_seeds, set_seeds, CustomDataset

def main(args):

    set_seeds(args.init_seed)

    all_dataset_names = args.dataset_name
    for i_data, dataset_name in enumerate(all_dataset_names):
        print(f'\nProcessing dataset: {dataset_name}')
        if dataset_name == 'cifar10-pretrained':
            dataset_name = 'cifar10'
            assert args.use_pretrained_features, 'Pretrained features must be used with cifar10-pretrained dataset'
        # draw rng seed for splitting
        seed = get_seeds(n_rng=1, initial_seed=args.init_seed)[0]
        print(f'Using seed: {seed}')

        if dataset_name in ['cifar10', 'fashion_mnist']:
            client_data, N_is, data_props = get_image_data(num_clients=args.num_clients+args.num_test_clients, dataset_name=dataset_name, data_folder=args.dataset_folder, random_seed=seed)
        elif dataset_name == 'income':
            # download and preprocess ACSIncome data
            print('Income data has inherent number of clients (51), ignoring num_clients and num_test_clients args')
            
            # some basic definitions for handling ACSIncome data, see https://github.com/socialfoundations/folktables
            ACSIncome_categories = {
                "COW": {
                    1.0: (
                        "Employee of a private for-profit company or"
                        "business, or of an individual, for wages,"
                        "salary, or commissions"
                    ),
                    2.0: (
                        "Employee of a private not-for-profit, tax-exempt,"
                        "or charitable organization"
                    ),
                    3.0: "Local government employee (city, county, etc.)",
                    4.0: "State government employee",
                    5.0: "Federal government employee",
                    6.0: (
                        "Self-employed in own not incorporated business,"
                        "professional practice, or farm"
                    ),
                    7.0: (
                        "Self-employed in own incorporated business,"
                        "professional practice or farm"
                    ),
                    8.0: "Working without pay in family business or farm",
                    9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
                },
                "SCHL": {
                    1.0: "No schooling completed",
                    2.0: "Nursery school, preschool",
                    3.0: "Kindergarten",
                    4.0: "Grade 1",
                    5.0: "Grade 2",
                    6.0: "Grade 3",
                    7.0: "Grade 4",
                    8.0: "Grade 5",
                    9.0: "Grade 6",
                    10.0: "Grade 7",
                    11.0: "Grade 8",
                    12.0: "Grade 9",
                    13.0: "Grade 10",
                    14.0: "Grade 11",
                    15.0: "12th grade - no diploma",
                    16.0: "Regular high school diploma",
                    17.0: "GED or alternative credential",
                    18.0: "Some college, but less than 1 year",
                    19.0: "1 or more years of college credit, no degree",
                    20.0: "Associate's degree",
                    21.0: "Bachelor's degree",
                    22.0: "Master's degree",
                    23.0: "Professional degree beyond a bachelor's degree",
                    24.0: "Doctorate degree",
                },
                "MAR": {
                    1.0: "Married",
                    2.0: "Widowed",
                    3.0: "Divorced",
                    4.0: "Separated",
                    5.0: "Never married or under 15 years old",
                },
                "SEX": {1.0: "Male", 2.0: "Female"},
                "RAC1P": {
                    1.0: "White alone",
                    2.0: "Black or African American alone",
                    3.0: "American Indian alone",
                    4.0: "Alaska Native alone",
                    5.0: (
                        "American Indian and Alaska Native tribes specified;"
                        "or American Indian or Alaska Native,"
                        "not specified and no other"
                    ),
                    6.0: "Asian alone",
                    7.0: "Native Hawaiian and Other Pacific Islander alone",
                    8.0: "Some Other Race alone",
                    9.0: "Two or More Races",
                },
            }
            states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
            # to get the state_sample_counts counts directly from folktables:
            """
            state_sample_counts = []
            for i_state, state in enumerate(states):
                tmp = data_source.get_data(states=[state], download=True)
                _, y , _ = ACSIncome.df_to_pandas(tmp, categories=ACSIncome_categories, dummies=True)
                state_sample_counts.append(len(y))
            del tmp, y
            """
            state_sample_counts = [22268, 3546, 33277, 13929, 195665, 31306, 19785, 4713, 98925, 50915, 7731, 8265, 67016, 35022, 17745, 15807, 22006, 20667, 7002, 33042, 40114, 50008, 31021, 13189, 31664, 5463, 10785, 14807, 7966, 47781, 8711, 103021, 52067, 4455, 62135, 17917, 21919, 68308, 5712, 24879, 4899, 34003, 135924, 16337, 3767, 46144, 39944, 8103, 32690, 3064, 9071]
            
            # get data from folktables
            data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
            all_states_data = data_source.get_data(states=states, download=True)
            all_states_features, all_states_labels, _ = ACSIncome.df_to_pandas(all_states_data, categories=ACSIncome_categories, dummies=True)

            all_x, all_y = all_states_features.to_numpy().copy(), all_states_labels.to_numpy().copy().reshape(-1)
            del all_states_data, all_states_features, all_states_labels, data_source
            ind = 0
            for i_client, state in enumerate(states):
                tmp = args.dataset_folder + f'/income/federated/inherent-clients51-testclients0/' + str(i_client)
                Path(tmp).mkdir(parents=True, exist_ok=True)
                ind_ = int(ind + state_sample_counts[i_client])
                # normalize first 5 cols which are continuous state-wise
                for i in range(5):
                    a = np.amin(all_x[ind:ind_,i])
                    b = np.amax(all_x[ind:ind_,i])
                    all_x[ind:ind_,i] = (all_x[ind:ind_,i] - a)/(b - a)
                # save client data
                np.savez_compressed(tmp + f'/client_{i_client}',x=all_x[ind:ind_,:], y=all_y[ind:ind_], allow_pickle=False)            
                
                ind += int(state_sample_counts[i_client])
            print(f'All client data for Income saved to {args.dataset_folder}/income/federated/inherent-clients51-testclients0/')
            return
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')
        
        # shuffle clients & split into train and test clients
        client_ids = np.arange(len(client_data),dtype=int)
        np.random.shuffle(client_ids)
        train_client_ids = client_ids[:args.num_clients]
        test_client_ids = client_ids[args.num_clients:]

        # transform into features from pretrained model
        if args.use_pretrained_features:
            model = get_model(model_name=args.pretrained_model_name, data_dims=args.data_dims)
            print('Loaded pretrained model')
            client_data = create_pretrained_features(client_data, model)
            
        # train/val set
        if args.use_pretrained_features:
            tmp = f"{args.dataset_folder}/{dataset_name}/federated/iid-pretrained-clients{args.num_clients}-testclients{args.num_test_clients}"
        else:
            tmp = f"{args.dataset_folder}/{dataset_name}/federated/iid-clients{args.num_clients}-testclients{args.num_test_clients}"
        
        print(f'Saving partitioned client data to folder \n{tmp} and \n{tmp}-testset')

        # save clients
        save_client_data([ client_data[i] for i in train_client_ids], folder_name=tmp, data_props=
                         [ data_props[i] for i in train_client_ids], N_is=[N_is[i] for i in train_client_ids])
        save_client_data([ client_data[i] for i in test_client_ids], folder_name=tmp+'-testset', data_props=
                         [ data_props[i] for i in test_client_ids], N_is=[N_is[i] for i in test_client_ids])

        print(f'Done processing dataset: {dataset_name}.')


def create_pretrained_features(client_data, model):
    """fun for transforming cifar10 data into data with features from pretrained resnext model and cifar10 labels
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # gpu is only used for extracting features from pretrained model
    model = model.to(device)
    client_features = []
    with torch.no_grad():
        print('Transforming data with features from pretrained model..')
        for i_client, data in tqdm(enumerate(client_data)):
            client_features.append([])
            dataset = CustomDataset(x=torch.from_numpy(np.array(data['x'], dtype=np.float32)), y=torch.from_numpy(np.array(data['y'], dtype=np.int64)), data_transforms=None)
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)
                f = model(x).cpu().numpy()
                client_features[-1].append(f)

            client_features[-1] = {'x' : np.concatenate(client_features[-1], axis=0), 'y': data['y']}
            print(f"Client {i_client} has transformed features done, shape {client_features[-1]['x'].shape}")

    print('All client data transformed')
    return client_features
    
if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")
    parser.add_argument('--pretrained_model_name', default='ResNext29', choices=['ResNext29'],help="Only used for processing CIFAR10 data; should be 'ResNext29'.")
    parser.add_argument('--data_dims', default=(3,32,32), type=tuple, help="Number of data dimensions. Used only with pretrained model. Should be (3,32,32) for CIFAR10.")
    parser.add_argument('--use_pretrained_features', default=False, action='store_true', help="Use pretrained model to extract features from image data sets.")
    parser.add_argument('--dataset_name', default="income", help="Name of the dataset.", choices=['fashion_mnist', 'cifar10-pretrained','income'])
    parser.add_argument('--dataset_folder', default='data', type=str, help='Main data folder, individual data sets will create subfolders under the main folder.')
    parser.add_argument('--num_clients', default=51, type=int, help='Number of clients for training/val.')
    parser.add_argument('--num_test_clients', default=0, type=int, help='Number of clients for testing, should be 0.')
    parser.add_argument('--init_seed', default=2303, type=int, help='Random seed for data splitting.')
    args = parser.parse_args()
    
    if type(args.dataset_name) == str:
        args.dataset_name = [args.dataset_name]

    main(args)
