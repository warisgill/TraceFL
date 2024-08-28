"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""



import torchvision.transforms as transforms
from medmnist import INFO
import medmnist
from datasets import Dataset, DatasetDict
import random
import logging
import torch
import torchvision
from collections import Counter
from tqdm import tqdm

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import tqdm
from transformers import AutoTokenizer, AutoProcessor

from flwr_datasets.partitioner import PathologicalPartitioner

from flwr_datasets.partitioner import ShardPartitioner
from functools import partial




def _get_medmnist(data_flag='pathmnist', download=True):
    info = INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    print(
        f"INFO: {info}, \n  n_channels: {n_channels},  \n n_classes: {n_classes}")

    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(
        split='train',  download=download)
    test_dataset = DataClass(
        split='test',  download=download)

    # Convert to Hugging Face Dataset format
    def medmnist_to_hf_dataset(medmnist_dataset):
        data_dict = {"image": [], "label": []}
        for pixels, label in medmnist_dataset:
            data_dict["image"].append(pixels)
            data_dict["label"].append(label.item())
        return Dataset.from_dict(data_dict)

    hf_train_dataset = medmnist_to_hf_dataset(train_dataset)
    hf_test_dataset = medmnist_to_hf_dataset(test_dataset)

    # Combine datasets into a single dataset with splits
    hf_dataset = DatasetDict({
        "train": hf_train_dataset,
        "test": hf_test_dataset
    })

    logging.info(f'conversion to hf_dataset done')
    return hf_dataset


def tokenize_function_factory(cfg):
    input_col_name = "content" if cfg.dname == "dbpedia_14" else "text"

    def _default_tokenize_function(examples):
        return tokenizer(examples[input_col_name], truncation=True, padding='max_length', max_length=128)

    def _yahoo_answers_tokenize_function(examples):
        examples['label'] = examples['topic']
        return tokenizer(examples['question_title'] + ' ' + examples['question_content'], truncation=True, padding='max_length', max_length=128)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     cfg.mname, trust_remote_code=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.add_special_tokens({
    #                                                                                                                    'pad_token': '[PAD]'})

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.mname, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if cfg.dname == "yahoo_answers_topics":
        return _yahoo_answers_tokenize_function
    return _default_tokenize_function


def train_test_transforms_factory(cfg):
    train_transforms = None
    test_transforms = None
    # image_processor = AutoImageProcessor.from_pretrained(cfg.mname)
    if cfg.dname == "cifar10":
        def apply_train_transformCifar(example):
            transform = Compose([
                Resize((32, 32)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010))
            ])
            example['pixel_values'] = [
                transform(image.convert("RGB")) for image in example['img']]
            example['label'] = torch.tensor(example['label'])
            del example['img']
            return example

        def apply_test_transformCifar(example):
            transform = Compose([
                Resize((32, 32)),
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010))
            ])

            example['pixel_values'] = [
                transform(image.convert("RGB")) for image in example['img']]
            example['label'] = torch.tensor(example['label'])
            del example['img']
            return example

        train_transforms = apply_train_transformCifar
        test_transforms = apply_test_transformCifar
    elif cfg.dname == "mnist":
        def apply_train_transformMnist(example):

            transform = Compose([
                Resize((32, 32)),
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
            example['pixel_values'] = [
                transform(image.convert("RGB")) for image in example['image']]
            # example['pixel_values'] = transform(example['image'].convert("RGB"))
            example['label'] = torch.tensor(example['label'])
            return example

        def apply_test_transformMnist(example):
            transform = Compose([
                Resize((32, 32)),
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])

            example['pixel_values'] = [
                transform(image.convert("RGB")) for image in example['image']]
            # example['pixel_values'] = transform(
            #     example['image'].convert("RGB"))
            example['label'] = torch.tensor(example['label'])
            del example['image']

            return example

        train_transforms = apply_train_transformMnist
        test_transforms = apply_test_transformMnist
    elif cfg.dname in ['pathmnist', 'organamnist']:
        tfms = transforms.Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize(mean=[.5], std=[.5])
        ])

        def apply_transform(example):
            example['pixel_values'] = [
                tfms(image.convert('RGB')) for image in example['image']]
            example['label'] = torch.tensor(example['label'])
            del example['image']
            return example
        return {'train': apply_transform, 'test': apply_transform}

    else:
        raise ValueError(f"Unknown dataset: {cfg.dname}")

    return {'train': train_transforms, 'test': test_transforms}


# def getAudioTransform(cfg, max_duration=1.0):
#     # Load the feature extractor
#     feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.mname)

#     # Preprocessing function
#     def preprocess_function(examples):
#         audio_arrays = [x["array"] for x in examples["audio"]]
#         inputs = feature_extractor(
#             audio_arrays,
#             sampling_rate=feature_extractor.sampling_rate,
#             max_length=int(feature_extractor.sampling_rate * max_duration),
#             truncation=True,
#         )
#         return inputs

#     return preprocess_function


def getAudioTransforms(cfg, max_duration=1.0):
    processor = AutoProcessor.from_pretrained(cfg.mname)
    feature_extractor = processor.feature_extractor

    def preprocessFunctionGeneric(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        int(feature_extractor.sampling_rate * max_duration)

        processed_inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        processed_inputs["label"] = examples["label"]
        return processed_inputs

    def preprocessFunctionWhisper(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        int(feature_extractor.sampling_rate * max_duration)

        processed_inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        processed_inputs["label"] = examples["label"]
        return processed_inputs

    if cfg.mname == "openai/whisper-tiny":
        return {'train': preprocessFunctionWhisper, 'test': preprocessFunctionWhisper}
    else:
        return {'train': preprocessFunctionGeneric, 'test': preprocessFunctionGeneric}


def _initialize_image_dataset(cfg, dat_partitioner_func, fetch_only_test_data):
    target_label_col = "label"

    d = dat_partitioner_func(cfg, target_label_col, fetch_only_test_data)
    transforms = train_test_transforms_factory(cfg=cfg)
    d['client2data'] = {k: v.map(
        transforms['train'], batched=True, batch_size=256, num_proc=8).with_format("torch") for k, v in d['client2data'].items()}
    d['server_data'] = d['server_data'].map(
        transforms['test'], batched=True, batch_size=256, num_proc=8).with_format("torch")
    return d


def _initialize_transformer_dataset(cfg, dat_partitioner_func, fetch_only_test_data):
    target_label_col = "label"
    if cfg.dname == "yahoo_answers_topics":
        target_label_col = "topic"

    d = dat_partitioner_func(cfg, target_label_col, fetch_only_test_data)
    d['client2data'] = {k: v.map(
        tokenize_function_factory(cfg)) for k, v in d['client2data'].items()}
    d['server_data'] = d['server_data'].map(
        tokenize_function_factory(cfg))
    return d


def _initialize_audio_dataset(cfg, dat_partitioner_func, fetch_only_test_data):
    transforms_dict = getAudioTransforms(cfg)
    transform_train = transforms_dict['train']
    transform_test = transforms_dict['test']

    target_label_col = "label"
    logging.info(
        f'>> Audio dataset: {cfg.dname} and target_label_col: {target_label_col}')
    d = dat_partitioner_func(
        cfg, target_label_col, fetch_only_test_data, subtask="ks")
    # audio_transform = preprocess_audio

    d['client2data'] = {k: v.map(transform_train, remove_columns=[
                                 "audio", "file"], batched=True, batch_size=256, num_proc=8) for k, v in d['client2data'].items()}
    d['server_data'] = d['server_data'].map(transform_test, remove_columns=[
                                            "audio", "file"], batched=True, batch_size=256, num_proc=8)
    return d


def _load_dist_based_clients_server_datasets(cfg, dat_partitioner_func, fetch_only_test_data=False):
    """Load the dataset and return the dataload."""
    if cfg.dname in ["cifar10", "mnist", 'pathmnist', 'organamnist']:
        return _initialize_image_dataset(cfg, dat_partitioner_func, fetch_only_test_data)

    if cfg.dname in ['dbpedia_14', 'yahoo_answers_topics']:
        return _initialize_transformer_dataset(cfg, dat_partitioner_func, fetch_only_test_data)

    elif cfg.dname in ['speech_commands','superb']:
        return _initialize_audio_dataset(cfg, dat_partitioner_func, fetch_only_test_data)

    else:
        raise ValueError(f"Dataset {cfg.dname} not supported")


def getLabelsCount(partition, target_label_col):
    label2count = Counter(example[target_label_col]  # type: ignore
                          for example in partition)  # type: ignore

    return dict(label2count)


def _fix_partition(cfg, c_partition, target_label_col):
    label2count = getLabelsCount(c_partition, target_label_col)

    filtered_labels = {label: count for label,
                       count in label2count.items() if count >= 10}

    indices_to_select = [i for i, example in enumerate(
        c_partition) if example[target_label_col] in filtered_labels]  # type: ignore

    ds = c_partition.select(indices_to_select)

    assert cfg.max_per_client_data_size > 0, f"max_per_client_data_size: {cfg.max_per_client_data_size}"

    if len(ds) > cfg.max_per_client_data_size:
        # ds = ds.shuffle()
        ds = ds.select(range(cfg.max_per_client_data_size))

    if len(ds) % cfg.batch_size == 1:
        ds = ds.select(range(len(ds) - 1))

    partition_labels_count = getLabelsCount(ds, target_label_col)
    return {'partition': ds, 'partition_labels_count': partition_labels_count}


def _partition_helper(partitioner, cfg, target_label_col, fetch_only_test_data, subtask):
    # logging.info(f"Dataset name: {cfg.dname}")
    clients_class = []
    clients_data = []
    server_data = None
    fds = None
    if cfg.dname in ['pathmnist', 'organamnist']:
        hf_dataset = _get_medmnist(data_flag=cfg.dname, download=True)

        partitioner.dataset = hf_dataset['train']
        fds = partitioner
        
        logging.info(f'max data size {cfg.max_server_data_size}')

        if cfg.max_server_data_size < len(hf_dataset['test']):
            server_data = hf_dataset['test'].select(range(cfg.max_server_data_size))
        else:
            server_data = hf_dataset['test']

    # partition = partitioner.load_partition(partition_id=partition_id)
    # return partition
    else:
        if subtask is not None:
            fds = FederatedDataset(dataset=cfg.dname, partitioners={
                "train": partitioner}, subset=subtask)
        else:
            fds = FederatedDataset(dataset=cfg.dname, partitioners={
                "train": partitioner})

        server_data = fds.load_split("test").select(
            range(cfg.max_server_data_size))

    logging.info(
        f"Partition helper: Keys in the dataset are: {server_data[0].keys()}")

    for cid in range(cfg.num_clients):
        client_partition = fds.load_partition(cid)
        temp_dict = {}

        if cfg.max_per_client_data_size > 0:
            logging.info(f' Fixing partition for client {cid}')
            temp_dict = _fix_partition(cfg, client_partition, target_label_col)
        else:
            logging.info(f' No data partition fix requried for client {cid}')
            temp_dict = {'partition': client_partition, 'partition_labels_count': getLabelsCount(
                client_partition, target_label_col)}

        if len(temp_dict['partition']) >= cfg.batch_size:
            clients_data.append(temp_dict['partition'])
            clients_class.append(temp_dict['partition_labels_count'])

    logging.info(f" -- fix partition is done --")
    client2data = {f"{id}": v for id, v in enumerate(clients_data)}
    client2class = {f"{id}": v for id, v in enumerate(clients_class)}
    return {'client2data': client2data, 'server_data': server_data, 'client2class': client2class, 'fds': fds}


def _dirichlet_data_distribution(cfg, target_label_col, fetch_only_test_data, subtask=None):
    partitioner = DirichletPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        alpha=cfg.dirichlet_alpha,
        min_partition_size=0,
        self_balancing=True,
        shuffle=True,
    )

    return _partition_helper(partitioner, cfg, target_label_col, fetch_only_test_data, subtask)


def _sharded_data_distribution(num_classes_per_partition, cfg, target_label_col, fetch_only_test_data, subtask=None):
    partitioner = ShardPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        shard_size=2000,
        num_shards_per_partition=num_classes_per_partition,
        shuffle=True
    )
    return _partition_helper(partitioner, cfg, target_label_col, fetch_only_test_data, subtask)


def _pathological_partitioner(num_classes_per_partition, cfg, target_label_col, fetch_only_test_data, subtask=None):
    partitioner = PathologicalPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        num_classes_per_partition=num_classes_per_partition,
        shuffle=True,
        class_assignment_mode='deterministic'
    )
    return _partition_helper(partitioner, cfg, target_label_col, fetch_only_test_data, subtask)


class ClientsAndServerDatasets:
    """Prepare the clients and server datasets."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dist_partitioner_func = None
        self._set_distriubtion_partitioner()
        self._setup()

    def _set_distriubtion_partitioner(self):
        if self.cfg.data_dist.dist_type == 'non_iid_dirichlet':
            self.data_dist_partitioner_func = _dirichlet_data_distribution
        elif self.cfg.data_dist.dist_type == 'sharded-non-iid-1':
            self.data_dist_partitioner_func = partial(
                _sharded_data_distribution, 1)  # passing num_classes_per_partition
        elif self.cfg.data_dist.dist_type == 'sharded-non-iid-2':
            self.data_dist_partitioner_func = partial(
                _sharded_data_distribution, 2)
        elif self.cfg.data_dist.dist_type == 'sharded-non-iid-3':
            self.data_dist_partitioner_func = partial(
                _sharded_data_distribution, 3)
        elif self.cfg.data_dist.dist_type == 'PathologicalPartitioner-1':
            self.data_dist_partitioner_func = partial(
                _pathological_partitioner, 1)
        elif self.cfg.data_dist.dist_type == 'PathologicalPartitioner-2':
            self.data_dist_partitioner_func = partial(
                _pathological_partitioner, 2)
        elif self.cfg.data_dist.dist_type == 'PathologicalPartitioner-3':
            self.data_dist_partitioner_func = partial(
                _pathological_partitioner, 3)
        else:
            raise ValueError(
                f"Unknown distribution type: {self.cfg.data_dist.dist}")

    def _setup_hugging_dataset(self):
        d = _load_dist_based_clients_server_datasets(
            self.cfg.data_dist, self.data_dist_partitioner_func)
        self.client2data = d["client2data"]

        self.server_testdata = d["server_data"]
        self.client2class = d["client2class"]
        self.fds = d["fds"]
        logging.info(f"client2class: {self.client2class}")

        logging.info(f"> client2class {self.client2class}")

        data_per_client = [len(dl) for dl in self.client2data.values()]
        logging.info(f"Data per client in experiment {data_per_client}")
        min_data = min(len(dl) for dl in self.client2data.values())
        logging.info(f"Min data on a client: {min_data}")

    def _setup(self):
        self._setup_hugging_dataset()

    def get_data(self):
        """Return the clients and server data for simulation."""
        return {
            "server_testdata": self.server_testdata,
            "client2class": self.client2class,
            "client2data": self.client2data,
            "fds": self.fds
        }
