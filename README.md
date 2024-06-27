# conFEDential
conFEDential is a Python workbench to test the impact of different configurations on the performance of a Federated 
Learning (FL) model and to discover how resilient it is against Membership Inference Attacks (MIAs). A MIA is an attack
where the goal is to discover if a specific data point was used to train a model. Because of its simplicity, it is 
frequently used to evaluate the privacy of a model. They rely on the fact that  Machine Learning models often perform 
better on the data they have already seen. For instance, in the image below, it can be seen that a ResNet18 [[1]](#1) model 
trained on CIFAR10 [[2]](#2) has a lower loss on members than it has on non-members:
![img](https://i.imgur.com/s6YnQWq.png)
The project is made in mind to be as flexible as possible. Meaning, that nearly all inputs and configurations can be 
changed. This includes configuration of the (in)homogenous data, federation, victim model architecture, training 
hyperparameters, attack model and the attack model hyperparameters. For the attack, we use a modular version of the 
attack by Nasr et al. [[3]](#3). The attack is modular in the sense that the attack model automatically adjusts its 
expected input size to the victim model and its input data.

## Installation
The project was built using Python 3.10.7, but any later version should also work. If any problems occur, downgrade to
Python 3.10.7. To install the project, follow the steps below:

#TODO: Write this whilst setting up project on another laptop

#TODO: Make sure to write on how to modify the Flower code to have early stopping

## Experiment Configuration
In the `examples` folder, there are already several examples of attacks on different datasets. An experiment is defined
by a YAML file which contains the following keys with values of the corresponding type.
```yaml
simulation:
  data:
    dataset_name: str
    batch_size: int
    preprocess_fn: str
    splitter: None |
      alpha: float
      percent_non_iid: float
  federation:
    client_count: int
    fraction_fit: float
    local_rounds: int
  model:
    optimizer_name: Literal["FedAdam", "FedAvg", "FedNAG", "SingleLayerFedNL"]
    model_name: str
    criterion_name: str
    optimizer_parameters: dict
    model_architecture: List[dict] |
      repo_or_dir: str
      model: str
      out_features: int
attack:
  data_access: float
  message_access: Literal["client", "server-encrypted", "server"]
  aggregate_access: int | float | List[int]
  repetitions: int
  attack_simulation:
    batch_size: int
    optimizer_name: Literal["Adam", "FedAdam", "FedAvg", "FedNAG", "RMSprop"]
    optimizer_parameters: dict
    model_architecture:
      components:
        label: bool
        loss: bool
        activation: bool
        gradient: bool
        metrics: bool
      gradient_component: List[dict]
      fcn_component: List[dict]
      encoder_component: List[dict]
```
### Simulation
This part of the experiment-configuration describes the configurables of the victim model simulation: The data used for 
the simulation, the federation and the model that is trained. 
#### Data
The data value describes what dataset is used, how it is preprocessed, how it is split and in what batch size it is used.
- `dataset_name (str)`: The name of the dataset. This can be any dataset that is downloaded in the `.cache` folder or that is
locally available. By default, we support CIFAR10 [[2]](#2), CIFAR100[[2]](#2), MNIST [[4]](#4), Purchase100 [[5]](#5) 
and Texas100 [[5]](#5). See [contributing](#Contributing) on how to add more datasets.
- `batch_size (int)`: The batch size used for training the model.
- `preprocess_fn (str)`: An additional function that will be applied to the data. For instance, if the x value should be
divided by two, the function could be: 
```yaml
preprocess_fn: |
  def preprocess_fn(element):
    return {
        "x": element["x"] / 2,
        "y": element["y"]
    }
```
- `splitter`: The configuration that is used to split the data with the Dirichlet method using
[FedArtML](https://pypi.org/project/FedArtML/). It takes an attribute `alpha (float)` and `percent_non_iid (float)`. The 
`alpha` attribute is used to determine the concentration of the Dirichlet distribution. The `percent_non_iid` attribute 
is used to determine the percentage of non-iid data. If `splitter` is `None`, the data will be split iid, and each client 
gets the same amount of data.
#### Federation
The federation value describes what kind of federation wants to collaboratively train an ML model. It takes three 
values:
- `client_count (int)`: The amount of clients in the federation
- `fraction_fit (float)`: The ratio of clients that is selected per training round. For instance, if client_count is 100
and fraction_fit 0.1, 100 * 0.1 = 10 clients will be selected per round for local training.
- `local rounds (int)`: How many epochs each client does locally before returning the model.
#### Model
The model value describes the model architecture that is used and with which hyperparameters it is trained.
- `optimizer_name (Literal["FedAdam", "FedAvg", "FedNAG", "SingleLayerFedNL"])`: The optimization protocol that is 
followed by the clients to train the model. See [contributing](#Contributing) on how to add more optimisers.
- `model_name (str)`: How the model will be named in the W&B logs.
- `criterion_name (str)`: The name of the [PyTorch loss function](https://pytorch.org/docs/stable/nn.html#loss-functions)
that should be used to train the model.
- `optimizer_parameters (dict)`: The hyperparameters that are used for the local and global optimizer. To see what
protocol takes which parameters, take a look in the initialization method of the protocol's class in 
`src.training.learning_method`. For instance, FedAdam may take:
```yaml
optimizer_parameters:
  local:
    lr: 0.01
  global:
    lr: 0.01
    betas:
      - 0.9
      - 0.99
    eps: 0.0001
```
- `model_architecture (dict)`: A description of the model architecture that the federation uses, either self defined or
imported. If imported, make sure the model is downloaded with `download_model.py`. In that case, `model_architecture`
contains the same variables as [`torch.hub.load()`](https://pytorch.org/docs/stable/hub.html#torch.hub.load):
  - `repo_or_dir (str)`
  - `model (str)`:
  - `out_features (int)`: The amount of out features of the model

If the model is self defined, it should contain a list of dicts. Each dict describes a layer of the architecture.
The value `type` describes the kind of [`torch.nn`](https://pytorch.org/docs/stable/nn.html) layer, all other values
are regarded as parameters to that layer. For example:
```yaml
model_architecture:
  - type: Linear
    in_features: 784
    out_features: 10
```
### Attack
This part of the experiment-configuration describes the attack configurables: The strength of the attacker and what 
model they use. Its direct children are:
- `data_access (float)`: What fraction of total data the attacker can use to train their attacker model. If 1.0 the
attacker gets all data samples except the victim.
- `message_access (Literal["client", "server-encrypted", "server"])`: The access the attacker has on the individual
messages transmitted within the network. This results in the following access:
  - `"client"`: All the messages transmitted by one random client of the federation simulation.
  - `"server-encrypted"`: No messages; A way to simulate secure aggregation.
  - `"server"`: All the messages from all clients
- `aggregate_access (int | float | List[int])`: The model aggregates that are used to attack the system. No distinction is
made in what role the attacker has in the network, e.g. local or global. The value means the following for the different
types:
  - `int`: The n number of intercepted round. The latest n server aggregates will be used by the attacker.
  - `float`: The fraction of rounds to which the attacker has access. The latest rounds will be selected first.
  - `List[int]`: The specific round indices which will be used by the attacker. Round 0 means the initial parameters, 
  the first aggregate resides in index 1.
#### Attack Simulation
The hyperparameters that are used to train the attack model. The attack is always trained using 
[`torch.nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss). It takes the
following direct values:
- `batch_size (int)` The batch size with which the attack model will be trained. It is recommended to keep this low
since the input for the attack model is often (very) big.
- `optimizer_name (Literal["Adam", "FedAdam", "FedAvg", "FedNAG", "RMSprop"])` The optimizer that is used 
to compute the model update after a batch. See [contributing](#contributing) on how to add more optimizers.
- `optimizer_parameters` The hyperparameters that are used by the optimizer. See the `get_optimizer()` method in the
optimizer classes in `src.training.learning_method` to see which parameters each value takes.
##### Model Architecture
The model architecture describes the components that make up the attack model. It is composed of these
4 values.
###### Components
Marks which components should be included in the attack model (and thus computed and passed as input).
- `loss (bool)`: The loss value of the victim model prediction and true label
- `label (bool)`: The true label that the model _should_ predict
- `activation (bool)`: The forward-pass values of all layers of the victim model.
- `gradient (bool)`: The backward-pass values of all layers of the victim model.
- `metrics (bool)`: The update to the metric value of all layers of the victim model. The metric is the
additional information that is transmitted between client and server or vice versa that is used to speed up
training, e.g. velocity for FedNAG.
###### Gradient Component
Describes the component that the gradient and metric go through before being passed to the FCN. Should contain a list 
of dicts. Each dict describes a layer of the architecture. The value `type` describes the kind of 
[`torch.nn`](https://pytorch.org/docs/stable/nn.html) layer, all other values are regarded as parameters
to that layer. For example:
```yaml
model_architecture:
  - type: Conv2d
    out_channels: 1000
    kernel_size:
    stride: 1
``` 
The first convolutional layer should not contain a value for `kernel_size` and `in_channels`. These values
are automatically adjusted to fit to the victim model.
###### Fully Connected Component
Describes the component that all values go through before being passed to the encoder. Should contain a list 
of dicts. Each dict describes a layer of the architecture. The value `type` describes the kind of 
[`torch.nn`](https://pytorch.org/docs/stable/nn.html) layer, all other values are regarded as parameters
to that layer. For example:
```yaml
model_architecture:
  - type: Linear
    out_features: 128
``` 
The first linear layer should not contain a value for `in_features`. This value is automatically adjusted to fit
to the victim model.
###### Encoder Component
Takes the output of all components and processes them to a prediction of a singular value. Should contain a list 
of dicts. Each dict describes a layer of the architecture. The value `type` describes the kind of 
[`torch.nn`](https://pytorch.org/docs/stable/nn.html) layer, all other values are regarded as parameters
to that layer. For example:
```yaml
model_architecture:
  - type: Linear
    out_features: 1
``` 
The first linear layer should not contain a value for `in_features`. This value is automatically adjusted
to the victim model and the components that are considered.

### Batched experiment
Multiple experiments can be described with the experiment configuration. This can be done in three ways:
- Described with `values` followed by a list of values: If the `values` keyword is used, all other variables which have
`values` will be cross-examined. For instance these optimizer parameters will result in four experiments. Both notations
are valid YAML notations.
```yaml
optimizer_parameters:
  lr: 
    values:
    - 0.01
    - 0.1
  momentum:
    values: [0.9, 0.95]
```
- Described with `min` `max` and `step_size`: If the `min` keyword is found, it will compute all the values between 
(incl bounds) `min` and `max` with an interval of `step_size`. Works with `values`. The following results in 6
experiments, lr specifically takes on `[0.1, 0.1, 0.2, 0.2, 0.3, 0.3]`
```yaml
optimizer_parameters:
  lr: 
    min: 0.1
    max: 0.3
    step_size: 0.1
  momentum:
    values: [0.9, 0.95]
```
- Described with `step`. All the indices of `step` will be taken together. So, if multiple variables declare `step`,
all values should be of the same length. Does not work with `min` or `values`. For the following experiment, 2
experiments will run with the config `[{lr: 0.01, momentum: 0.9}, {lr: 0.1, momentum: 0.95}]`:
```yaml
optimizer_parameters:
  lr: 
    steps:
    - 0.01
    - 0.1
  momentum:
    steps: [0.9, 0.95]
```

## Intermediate Results
To save resources, expensive computations are stored for later use. In the `.cache` folder, we store two folders the 
`data` and `model_architectures` folder. The `model_architectures` folder used to store downloaded model architectures.
The `data` folder stores per dataset intermediate computations. For instance, for the `purchase` dataset. In this folder
3 more folders are present: `preprocessed`, `training` and some other folder in which the raw data sits. 

`preprocessed` contains the split and unsplit preprocessed data. The SHA256 hash of the preprocess function is taken
as directory name. This directory contains an `unsplit.pkl` pickle file and a `split` directory. The files in `split`
are the dataloaders for a federation of `client_count` with a `batch_size`. These variables are put in a dict and that
hash is the filename of the stored dataloader.

`training` contains the trained model of a simulation. First on `model_name`, then on optimization protocol and then on
the hash of the entire simulation configuration. This directory contains the `aggregates` and the individual 
`messages` of the protocol. Both contain either `aggregates.hdf5` or `messages.hdf5` in which the aggregates
or messages of the model parameters are stored. The extra information that is sent from the server to client
(aggregates) or from client to server (messages) is stored in the `metrics` directory, where the filename
is the variable name of the metric.

This results in the following file structure for `.cache/data/purchase`:
```
.cache/data/purchase/
|   preprocessed
|   |   6953241a9b8ff95102b78518189c2b990449e30cf33ece85878cd0da48066bd8
|   |   |   split
|   |   |   |   0c1a3691636445c7c3a31a8ee544cd4d81d9a5405ab86929f9389d9a2acf45a3.pkl
|   |   |   |   3135f4a9fd34f3f9a9034f75ff877a4639f862996466129c434222b61345c35b.pkl
|   |   |   unsplit.pkl
|   purchase
|   |   purchase.parquet
|   training
|   |   FCN
|   |   |   FedNAG
|   |   |   |   c2b404ee41277a6afdfab9490665b7aefc59a679e922532f758ee99a9acaec6f
|   |   |   |   |   aggregates
|   |   |   |   |   |   aggregates.hdf5
|   |   |   |   |   |   metrics
|   |   |   |   |   |   |   velocity.hdf5
|   |   |   |   |   messages
|   |   |   |   |   |   messages.hdf5
|   |   |   |   |   |   metrics
|   |   |   |   |   |   |   velocity.hdf5
```


## Contributing
### Adding Datasets
Besides CIFAR10, CIFAR100, MNIST, Purchase100 and Texas100, other datasets can be added. For that, you need to follow
these steps:
1. Download the dataset in `.cache/data/{dataset}/` where `{dataset}` is the name of the dataset.
2. Create a class in `src.data` for that dataset and export it in the `__init__` file of the module. It is recommended
to also export some aliases in the `__init__` file to account for capitalisation errors in the experiment config.
3. Make sure the new class implements `src.data.Dataset` and implement the `load_dataset` method. This function should
check whether the dataset is downloaded locally before doing anything else. This is done by calling 
`Dataset.is_data_downloaded({dataset}, cache_root)`. The function should return 3 HuggingFace datasets:
train, test and validation. For information on how to import datasets with HuggingFace, [see this link](https://huggingface.co/docs/datasets/loading).

### Adding Optimisers
Besides the FedAdam [[6]](#6), FedAvg [[7]](#7), FedNAG [[8]](#8) and FedNL [[9]](#9) protocols, more protocols can be 
added. For that, you need to follow these steps:
1. Create a class in `src.training.learning_methods` with the protocol name and export it in the `__init__` file of 
the module. It is recommended to also export some aliases in the `__init__` file to account for capitalisation errors in 
the experiment config.
2. Make sure the new class implements `src.training.learning_methods.Strategy`. Like `Adam` or `RMSprop`, not all 
methods do need to be implemented. Those only have `get_optimizer` as the optimizer function was only needed for their
purpose. The methods do the following:
   1. `get_optimizer`: returns the optimizer that the client uses locally to train on.
   2. `train`: the algorithm that the client runs locally to train on a given train dataloader. Should return
   the new parameters, the amount of data, and any metrics.
   3. `aggregate_fit`: the algorithm that the server runs to combine the results of the clients. Should return
   the globally new parameters and any aggregated metrics. The results of `aggregate_fit` are captured.
   4. `get_server_exclusive_metrics`: any keys of metrics that are exclusive to the server. Those values are not
   given to a local attacker.
   5. `compute_metric_updates`: a method that computes the metric update for a set of features over the
   different received models using the initial metric.
 
## References
<a id="1">[1]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual
Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR). 770–778. https://doi.org/10.1109/CVPR.2016.90

<a id="2">[2]</a> Alex Krizhevsky. 2009. Learning multiple layers of features from tiny images.
Technical Report.

<a id="3">[3]</a> Milad Nasr, Reza Shokri, and Amir Houmansadr. 2019. Comprehensive Privacy
Analysis of Deep Learning: Passive and Active White-box Inference Attacks
against Centralized and Federated Learning. In 2019 IEEE Symposium on Security
and Privacy (SP). 739–753. https://doi.org/10.1109/SP.2019.00065

<a id="4">[4]</a> Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. 
IEEE Signal Processing Magazine, 29(6), 141–142.

<a id="5">[5]</a> Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Mem-
bership Inference Attacks Against Machine Learning Models. In 2017 IEEE Sym-
posium on Security and Privacy (SP). 3–18. https://doi.org/10.1109/SP.2017.41

<a id="6">[6]</a> Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
Jakub Konečný, Sanjiv Kumar, and H. Brendan McMahan. 2021. Adaptive Feder-
ated Optimization. arXiv:2003.00295 (Sept. 2021). https://doi.org/10.48550/arXiv.
2003.00295 arXiv:2003.00295 [cs, math, stat].

<a id="7">[7]</a> H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise
Agüera y Arcas. 2023. Communication-Efficient Learning of Deep Networks from
Decentralized Data. arXiv:1602.05629 (Jan. 2023). http://arxiv.org/abs/1602.05629
arXiv:1602.05629 [cs].

<a id="8">[8]</a> Zhengjie Yang, Wei Bao, Dong Yuan, Nguyen H. Tran, and Albert Y. Zomaya. 2022. Federated Learning with Nesterov Accelerated Gradient. IEEE Transactions
on Parallel and Distributed Systems 33, 12 (Dec. 2022), 4863–4873. https://doi.org/
10.1109/TPDS.2022.3206480 arXiv:2009.08716 [cs, stat].

<a id="9">[9]</a> Mher Safaryan, Rustem Islamov, Xun Qian, and Peter Richtárik. 2022. FedNL: Mak-
ing Newton-Type Methods Applicable to Federated Learning. arXiv:2106.02969
(May 2022). https://doi.org/10.48550/arXiv.2106.02969 arXiv:2106.02969 [cs,
math].