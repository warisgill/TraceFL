# TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance

> **Accepted at 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)** [[Arxiv Paper Link](https://arxiv.org/pdf/2312.13632)]

For questions or feedback, please contact at [waris@vt.edu](mailto:waris@vt.edu). The `code` is written in [Flower FL Framework](https://flower.ai/), the most widely used FL framework.
# 1. TraceFL
TraceFL is a `tool` designed to provide **interpretability** in Federated Learning (FL) by identifying clients responsible for specific predictions made by a global model.

![alt text](image.png)
### 1.1 Overview
Federated Learning (FL) enables multiple clients (e.g., `hospitals` ) to collaboratively train a global model without sharing their raw data. However, this distributed and privacy-preserving setup makes it challenging to attribute a model's predictions to specific clients. Understanding which clients are most responsible for a model's output is crucial for `debugging`, `accountability`, and `incentivizing` high-quality contributions.

TraceFL addresses this challenge by dynamically tracking the significance of neurons in a global model's prediction and mapping them back to the corresponding neurons in each participating client's model. This process allows FL developers to localize the clients most responsible for a prediction without accessing their raw training data.
### 1.2 Key Features
- **Neuron Provenance:** A novel technique that tracks the flow of information from individual clients to the global model, identifying the most influential clients for each prediction.
- **High Accuracy:** TraceFL achieves 99% accuracy in localizing responsible clients in both image and text classification tasks.
- **Wide Applicability:** Supports multiple neural network architectures, including CNNs (e.g., ResNet, DenseNet) and any transformers model from HuggingFace library (e.g., BERT, GPT).
- **Scalability and Robustness:** Efficiently scales to thousands of clients and maintains high accuracy under varying data distributions and differential privacy settings.
- **No Client-Side Instrumentation Required:** Runs entirely on the central server, without needing access to clients' training data or modifications to the underlying fusion algorithm.
# 2. Running TraceFL

>The `.sh` (e.g., `job_training_all_exps.sh`) scripts and `TraceFL/tracefl/conf/base.yaml` provided in this artifact can be used to regenerate any experiment results presented in the paper. `

The experiments cover various aspects of federated learning, including:
1. **Image and Text Classification**: Evaluating the performance of different models and datasets in federated settings.
2. **Differential Privacy**: Analyzing the impact of differential privacy on model training and TraceFL's localizability.
3. **Scalability**: Testing the scalability of TraceFL with varying numbers of clients and rounds.
4. **Dirichlet Alpha Tuning**: Exploring the effects of different Dirichlet alpha values on data distribution, TraceFL's localizability, and model performance.
### 2.1 Experiments Configuration Overview
- **Image Classification**:
  - Models: ResNet18, DenseNet121
  - Datasets: MNIST, CIFAR-10, PathMNIST, OrganAMNIST
  - Number of Rounds: 25-50
- **Text Classification**:
  - Models: OpenAI GPT, Google BERT
  - Datasets: DBPedia, Yahoo Answers
  - Number of Rounds: 25
### 2.2 Differential Privacy Analysis
These experiments evaluate the impact of differential privacy on TraceFL by applying different noise levels and clipping norms.
- **Models**: DenseNet121, OpenAI GPT
- **Datasets**: MNIST, PathMNIST, DBPedia
- **Noise Levels**: 0.0001, 0.0003, 0.0007, 0.0009, 0.001, 0.003
- **Clipping Norms**: 15, 50
- **Number of Rounds**: 15
### 2.3 Scalability Experiments
Scalability tests involve running experiments with varying numbers of clients and rounds to assess how well TraceFL scales.
- **Models**: OpenAI GPT
- **Dataset**: DBPedia
- **Number of Clients**: 200, 400, 600, 800, 1000
- **Clients per Round**: 10, 20, 30, 40, 50
- **Number of Rounds**: 15, 100

### 2.4 Dirichlet Alpha Experiments
These experiments explore the effect of different Dirichlet alpha values on data partitioning,  model training, and TraceFL's localizability.
- **Models**: OpenAI GPT, DenseNet121
- **Datasets**: Yahoo Answers, DBPedia, PathMNIST, OrganAMNIST, MNIST, CIFAR-10
- **Dirichlet Alpha Values**: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
- **Number of Clients**: 100
- **Clients per Round**: 10
- **Number of Rounds**: 15
### 2.5 Results and Log Files
Each experiment's output will be logged in the `logs` directory, providing detailed information about the training process and results.

# 3. Potential Use Cases of TraceFL
- **Debugging and Fault Localization:** Identify and isolate faulty or malicious clients responsible for incorrect or suspicious predictions in federated learning models.
- **Enhancing Model Quality, Fairness, and Incentivization:**  Improve model performance by rewarding high-quality clients, ensuring fair client contributions, and incentivizing continued participation from beneficial clients.
- **Client Accountability and Security:** Increase accountability by tracing model decisions back to specific clients, deterring malicious behavior, and ensuring secure contributions.
-  **Optimized Client Selection and Efficiency:** Dynamically select the most beneficial clients for training to enhance model performance and reduce communication overhead.
- **Interpretable Federated Learning in Sensitive Domains:** Provide transparency and interpretability in federated learning models, crucial for compliance, trust, and ethical considerations in domains like healthcare and finance.

## 4. Citation
Latex
```
@inproceedings{gill2025tracefl,
  title = {{TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance}},
  author = {Gill, Waris and Anwar, Ali and Gulzar, Muhammad Ali},
  booktitle = {2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)},
  year = {2025},
  organization = {IEEE},
}
```

