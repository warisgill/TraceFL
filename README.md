## TraceFL: Localizing Responsible Clients in Federated Learning Systems

The `.sh` scripts provided in this artifact can be used to regenerate any experiment results presented in the paper. The experiments cover various aspects of federated learning, including:

1. **Image and Text Classification**: Evaluating the performance of different models and datasets in federated settings.
2. **Differential Privacy**: Analyzing the impact of differential privacy on model training and TraceFL's localizability.
3. **Scalability**: Testing the scalability of TraceFL with varying numbers of clients and rounds.
4. **Dirichlet Alpha Tuning**: Exploring the effects of different Dirichlet alpha values on data distribution, TraceFL's localizability, and model performance.

### Experiments Configuration Overview

- **Image Classification**:
  - Models: ResNet18, DenseNet121
  - Datasets: MNIST, CIFAR-10, PathMNIST, OrganAMNIST
  - Number of Rounds: 25-50

- **Text Classification**:
  - Models: OpenAI GPT, Google BERT
  - Datasets: DBPedia, Yahoo Answers
  - Number of Rounds: 25

#### 2. Differential Privacy Analysis

These experiments evaluate the impact of differential privacy on TraceFL by applying different noise levels and clipping norms.

- **Models**: DenseNet121, OpenAI GPT
- **Datasets**: MNIST, PathMNIST, DBPedia
- **Noise Levels**: 0.0001, 0.0003, 0.0007, 0.0009, 0.001, 0.003
- **Clipping Norms**: 15, 50
- **Number of Rounds**: 15

#### 3. Scalability Experiments

Scalability tests involve running experiments with varying numbers of clients and rounds to assess how well TraceFL scales.

- **Models**: OpenAI GPT
- **Dataset**: DBPedia
- **Number of Clients**: 200, 400, 600, 800, 1000
- **Clients per Round**: 10, 20, 30, 40, 50
- **Number of Rounds**: 15, 100

#### 4. Dirichlet Alpha Experiments

These experiments explore the effect of different Dirichlet alpha values on data partitioning,  model training, and TraceFL's localizability.

- **Models**: OpenAI GPT, DenseNet121
- **Datasets**: Yahoo Answers, DBPedia, PathMNIST, OrganAMNIST, MNIST, CIFAR-10
- **Dirichlet Alpha Values**: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
- **Number of Clients**: 100
- **Clients per Round**: 10
- **Number of Rounds**: 15

### Log Files

Each experiment's output will be logged in the `logs` directory, providing detailed information about the training process and results.

By following these steps, you can replicate and analyze the TraceFL experiments.
