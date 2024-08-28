"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""





import logging
from tracefl.dataset_preparation import ClientsAndServerDatasets, getLabelsCount
from diskcache import Index
from flwr_datasets.visualization import plot_label_distributions
import matplotlib.pyplot as plt
from pathvalidate import sanitize_filename
import numpy as np
import collections



def mdedical_dataset2labels(dname):
    if dname == 'pathmnist':
        return {0: 'Adipose', 1: 'Background', 2: 'Debris', 3: 'Lymphocytes', 4: 'Mucus', 5: 'Smooth Muscle', 6: 'Normal Colon Mucosa', 7: 'Cancer-associated Stroma', 8: 'Colorectal Adenocarcinoma'}
    else:
        # raise ValueError(f"Unknown dataset {dname}") 
        return None




def _save_graph_of_data_distribution(fds, fname, target_label_col= 'label') -> None:
    partitioner = fds.partitioners["train"]
    fig, ax, df = plot_label_distributions(
        partitioner,
        label_name=target_label_col,
        plot_type="heatmap",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
        plot_kwargs={"annot": True},
    )
    print(df)
    plt.savefig(f"{fname}.png")
    plt.close()
    df.to_csv(f"{fname}.csv")


def get_clients_server_data(cfg):
    ds_dict = {}
    cache_path = cfg.storage.dir + cfg.storage.fl_datasets_cache
    cache = Index(cache_path)

    dataset_key = f"-"
    for k, v in cfg.data_dist.items():
        dataset_key += f"{k}:{v}-"

    if dataset_key in cache.keys()  and cfg.check_dataset_cache:
        logging.warning(
            f"\nLoading dataset from cache {cache_path}: {dataset_key}\n")
        ds_dict   = cache[dataset_key]       
    else:
        ds_prep = ClientsAndServerDatasets(cfg)
        ds_dict = ds_prep.get_data()
        cache[dataset_key] = ds_dict
        logging.info(f"Saving dataset to cache {cache_path}: {dataset_key}")
    
    # _save_graph_of_data_distribution(ds_dict['fds'],fname= f"graphs/temp/{sanitize_filename(dataset_key)}")
    return ds_dict


def load_central_server_test_data(cfg):
    """Load the central server test data."""
    d_obj = ClientsAndServerDatasets(cfg).get_data()
    return d_obj["server_testdata"]






def convert_client2_faulty_client(ds, label2flip, target_label_col= 'label'):

    def flip_label(example):
        label = None
        try:
            label = example[target_label_col].item()
        except:
            label = example[target_label_col]
        if label in label2flip:
            example[target_label_col] = label2flip[label]  
        return example
    
    

    ds =  ds.map(flip_label).with_format("torch")

    label2count = {}

    for example in ds:
        label = example[target_label_col].item()
        # print(f'---> label {label}')
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1

    return {'ds': ds, 'label2count' : label2count}






