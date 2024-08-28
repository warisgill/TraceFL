"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import gc
import logging
import random
import time
import copy

import flwr as fl
import hydra
import torch
from diskcache import Index
from flwr.common import ndarrays_to_parameters

from tracefl.client import FlowerClient, get_parameters, set_parameters
from tracefl.dataset import get_clients_server_data, mdedical_dataset2labels, convert_client2_faulty_client


from tracefl.differential_testing import run_fed_debug_differential_testing
from tracefl.models import global_model_eval, initialize_model
from tracefl.strategy import FedAvgSave
from tracefl.utils import set_exp_key 
from tracefl.fl_provenance import given_key_provenance, run_full_cache_provenance
from tracefl.plotting import do_plotting,convert_cache_to_csv, plot_label_distribution
from tracefl.differential_testing import run_fed_debug_differential_testing

import multiprocessing

# from collections import Counter








class FLSimulation:
    """Main class to run the simulation."""

    def __init__(self, cfg, cache):
        self.all_rounds_results = []
        self.cache = cache
        self.cfg = cfg
        self.strategy = None
        self.device = torch.device(self.cfg.device)

        self.client_resources = {"num_cpus": cfg.client_cpus}
        if self.cfg.device == "cuda":
            self.client_resources = {
                "num_gpus": cfg.client_gpu,
                "num_cpus": cfg.client_cpus,
            }

        init_args = {"num_cpus": self.cfg.total_cpus,
                     "num_gpus": self.cfg.total_gpus}
        self.backend_config = {
            "client_resources": self.client_resources,
            "init_args": init_args,
        }

    def set_server_data(self, ds):
        """Set the server data."""
        self.server_testdata = ds

    def set_clients_data(self, c2d):
        """Set the clients data."""
        self.client2data = c2d
        logging.info(f'Number of clients: {len(self.client2data)}')

        logging.info(
            f'Participating clients ids  are {list(self.client2data.keys())}')

        if len(self.client2data) != self.cfg.num_clients:
            logging.warning(
                f"Number of clients in client2data: {len(self.client2data)} is not equal to the number of clients specified in the config: {self.cfg.data_dist.num_clients}")

            # reducing number of clients
            self.cfg.num_clients = len(self.client2data)
            logging.warning(
                f"Reducing number of clients to: {self.cfg.data_dist.num_clients}")

    def set_strategy(self):
        initial_net = initialize_model(
            self.cfg.model.name, self.cfg.dataset)["model"]
        if self.cfg.strategy.name in ["fedavg"]:
            strategy = FedAvgSave(
                cfg=self.cfg,
                cache=self.cache,
                accept_failures=False,
                fraction_fit=0,
                fraction_evaluate=0.0,
                min_fit_clients=self.cfg.strategy.clients_per_round,
                min_evaluate_clients=0,
                min_available_clients=self.cfg.data_dist.num_clients,
                initial_parameters=ndarrays_to_parameters(
                    ndarrays=get_parameters(initial_net)
                ),
                evaluate_fn=self._evaluate_global_model,  # ignore
                on_fit_config_fn=self._get_fit_config,  # Pass the fit_config function
                fit_metrics_aggregation_fn=self._fit_metrics_aggregation_fn,
            )

            if self.cfg.strategy.noise_multiplier != -1 and self.cfg.strategy.clipping_norm != -1:
                logging.info(
                    ">> ----------------------------- Running DP FL -----------------------------")
                dp_strategy = fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(
                    strategy, noise_multiplier=self.cfg.strategy.noise_multiplier, clipping_norm=self.cfg.strategy.clipping_norm, num_sampled_clients=self.cfg.strategy.clients_per_round)
                self.strategy = dp_strategy
            else:
                logging.info(
                    f">> ----------------------------- Running Non-DP FL -----------------------------\n >> strategy_cfg: {self.cfg.strategy} ")

                self.strategy = strategy

    def _fit_metrics_aggregation_fn(self, metrics):
        """Aggregate metrics recieved from client."""
        logging.info(">>   ------------------- Clients Metrics ------------- ")
        all_logs = {}
        for t in metrics:
            nk, m = t
            cid = int(m["cid"])
            s = f' Client {m["cid"]}, Loss Train {m["train_loss"]}, Accuracy Train {m["train_accuracy"]}, data_points = {nk}'
            all_logs[cid] = s

        # sorted by client id from lowest to highest
        for k in sorted(all_logs.keys()):
            logging.info(all_logs[k])

        return {"loss": 0.1, "accuracy": 0.2}

    def _get_fit_config(self, server_round: int):
        random.seed(server_round)
        torch.manual_seed(server_round)
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": self.cfg.client.epochs,  #
            "batch_size": self.cfg.data_dist.batch_size,
            "lr": self.cfg.client.lr,
        }
        gc.collect()
        return config

    def _evaluate_global_model(self, server_round, parameters, config):
        gm_dict = initialize_model(self.cfg.model.name, self.cfg.dataset)
        set_parameters(gm_dict["model"], parameters)
        gm_dict["model"].eval()  # type: ignore
        d = global_model_eval(self.cfg.model.arch,
                              gm_dict, self.server_testdata)
        loss = d["loss"]
        accuracy = d["accuracy"]
        self.all_rounds_results.append({"loss": loss, "accuracy": accuracy})

        gc.collect()
        return loss, {"accuracy": accuracy, "loss": loss, "round": server_round}

    def _get_client(self, cid):
        model_dict = initialize_model(self.cfg.model.name, self.cfg.dataset)
        client = None

        args = {
            'arch': self.cfg.model.arch,
            "cid": cid,
            "model_dict": model_dict,
            "client_data_train": self.client2data[cid],
            "valloader": None,
            "device": self.device,
            "mode": self.cfg.strategy.name,
            'output_dir': self.cfg.storage.dir,
        }
        # print(f'args for client {cid}: {args}')

        client = FlowerClient(args).to_client()
        return client

    def run(self):
        """Run the simulation."""
        logging.info('Running the simulation')
        client_app = fl.client.ClientApp(client_fn=self._get_client)

        server_config = fl.server.ServerConfig(
            num_rounds=self.cfg.strategy.num_rounds)
        server_app = fl.server.ServerApp(
            config=server_config, strategy=self.strategy)

        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=self.cfg.num_clients,
            backend_config=self.backend_config,  # type: ignore
        )
        return self.all_rounds_results

    # def run(self):
    #     """Run the simulation."""
    #     fl.simulation.start_simulation(
    #         ray_init_args=self.backend_config["init_args"],
    #         client_fn=self._get_client,
    #         num_clients=self.cfg.num_clients,
    #         config=fl.server.ServerConfig(
    #             num_rounds=self.cfg.strategy.num_rounds
    #         ),  # Just three rounds
    #         strategy=self.strategy,
    #         client_resources=self.backend_config["client_resources"],
    #     )

    #     return self.all_rounds_results


def run_training_simulation(cfg):
    """Run the simulation."""
    train_cache = Index(cfg.storage.dir + cfg.storage.train_cache_name)
    if cfg.check_train_cache:
        if cfg.exp_key in train_cache:
            temp_dict = train_cache[cfg.exp_key]
            if "complete" in temp_dict and temp_dict["complete"] and len(temp_dict["all_ronuds_gm_results"]) > 0:
                # print(f'---------> {temp_dict["all_ronuds_gm_results"]}')
                logging.info(f"Training already completed: {cfg.exp_key}, len of results: {len(temp_dict['all_ronuds_gm_results'])}")
                return

    logging.info(
        f" ***********  Starting Experiment: {cfg.exp_key} ***************")
    # logging.info(f"Simulation Configuration: {cfg}")

    ds_dict = get_clients_server_data(cfg)

    train_cfg = copy.deepcopy(cfg)  # important

    # ds_dict["client2class"]

    logging.info(f'Client to class mapping: {ds_dict["client2class"]}')
    plot_label_distribution(ds_dict["client2class"], mdedical_dataset2labels(cfg.dataset.name), fname_type='without_flip')

    logging.info(f'faulty clients: {cfg.faulty_clients_ids}')
    if len(cfg.faulty_clients_ids) > 0:
        cfg.faulty_clients_ids = [f"{x}" for x in cfg.faulty_clients_ids]
        logging.info(f'Converting clients to faulty clients: {cfg.faulty_clients_ids}')
        # _ = input("Press Enter to continue...")
        for faulty_id in cfg.faulty_clients_ids:
            client_new_ds_dict  = convert_client2_faulty_client(ds_dict["client2data"][faulty_id], cfg.label2flip)
            ds_dict["client2data"][faulty_id], label2count = client_new_ds_dict['ds'], client_new_ds_dict['label2count']
            temp_client2class = copy.deepcopy(ds_dict["client2class"])   
            temp_client2class[faulty_id] = label2count
        logging.info(f'After Client to class mapping: {temp_client2class}')
        plot_label_distribution(temp_client2class, mdedical_dataset2labels(cfg.dataset.name), fname_type='flip')
    
    # _ = input("Press Enter to continue...")

    sim = FLSimulation(copy.deepcopy(cfg), train_cache)
    sim.set_server_data(ds_dict["server_testdata"])
    sim.set_clients_data(ds_dict["client2data"])
    sim.set_strategy()
    round2results = sim.run()

    temp_input_shape = None

    if cfg.model.arch == "cnn":
        if "pixel_values" in sim.server_testdata[0]:
            temp_input_shape = sim.server_testdata[0]["pixel_values"].clone(
            ).detach().shape
        else:
            temp_input_shape = sim.server_testdata[0][0].clone().detach().shape

    train_cache[cfg.exp_key] = {
        "client2class": ds_dict["client2class"],
        "train_cfg": train_cfg,
        "complete": True,
        "input_shape": temp_input_shape,
        "all_ronuds_gm_results": round2results,
    }

    logging.info(f"Results of gm evaluations each round: {round2results}")
    logging.info(f"Training Complete for: {cfg.exp_key} ")

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()





@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg) -> None:
    """Run the baseline."""

    cfg.exp_key = set_exp_key(cfg)
    if cfg.dry_run:
        logging.info(f"DRY RUN: {cfg.exp_key}")
        return

    if cfg.do_training:
        # try:
        run_training_simulation(cfg)
        # except Exception as e:
        #     logging.error(f"Error: {e}")
        #     logging.error(f"Error in training experiment: {cfg.exp_key}")
        # time.sleep(1)


    if cfg.do_provenance:
        # multiprocessing.set_start_method('spawn')
        logging.info("Running Provenance")
        given_key_provenance(cfg)
        # run_fed_debug_differential_testing(cfg)

        # try:
        #     given_key_provenance(cfg)
        # except Exception as e:
        #     logging.error(f"Error: {e}")
        #     logging.error(f"Error in provenance experiment: {cfg.exp_key}")
        
    
    # if cfg.do_full_cache_provenance:
    #     run_full_cache_provenance(cfg)

    if cfg.convert_cache_to_csv:
        convert_cache_to_csv(Index(cfg.storage.dir + cfg.storage.results_cache_name))
        logging.info("Converted cache to csv is done")
    
    if cfg.plotting:
        logging.info("Plotting results")
        do_plotting()


if __name__ == "__main__":
    main()
