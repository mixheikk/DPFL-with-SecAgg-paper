### Setup

1) Install [poetry](https://python-poetry.org/docs/), the code has been tested with Python version 3.10.14.

2) Run the following code in the folder containing pyproject.toml to install modified Flower and all dependencies (the tested code simulates FL clients without running on actual devices)

```
poetry install --extras simulation
```

3) Download the pre-trained [ResNeXt-29 8x64 weights](https://github.com/bearpaw/pytorch-classification), and point torch.load to the correct file (line 22 in models.py).

### For running the code:

i) Privacy accounting should be done as a separate step before running the main file to get the correct noise level. Check the argparser options in dp_accounting.py for details.

ii) Client datasets are created by create_client_data.py. Check the argparser options at the end of the file before running.

iii) The main file for running all experiments is federated_learning_main.py. Before running, check the (longish) list of argparser options at the end of the file to set the configuration. To run experiments with fixed configurations used in the paper, utilise the provided config files (giving config file name for argparser overwrites the corresponding default options). For example, for making a single run with the same hyperparameters as in the one local step experiment with [ACS Income data](https://github.com/socialfoundations/folktables) in the paper (see Fig. 3), run the following commands:

```
python create_client_data.py --dataset_name income
python federated_learning_main.py --config config-files/income-skellam-1step-inherent-q05.yaml
```

### Code Acknowledgements

The codebase is built on the [Flower federated learning framework](https://github.com/adap/flower). The Skellam mechanism implementation and accounting is based on the code from https://github.com/facebookresearch/dp_compression. The code used for preprocessing ACS Income data is based on the original [Ding et al. 2021 paper code](https://github.com/socialfoundations/folktables) as well as on preprocessing code by [Luca Corbucci](https://github.com/lucacorbucci).