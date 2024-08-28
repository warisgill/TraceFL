import logging
import time
from diskcache import Index
from multiprocessing import Pool
from tracefl.dataset import get_clients_server_data
from tracefl.fl_provenance_modules import FederatedProvTrue
from tracefl.utils import get_prov_eval_metrics
from tracefl.models import test_neural_network
from tracefl.neuron_provenance import NeuronProvenance
import torch 
from tracefl.differential_testing import round_lambda_fed_debug_func
from joblib import Parallel, delayed

class FederatedProvFalse(FederatedProvTrue):
    def __init__(self, cfg, round_key, central_test_data, t=None):
        super().__init__(cfg, round_key, central_test_data, t)
    

    def _sanityCheck(self):
        if len(self.subset_test_data) == 0:
            return None

        acc = test_neural_network(self.train_cfg.model.arch, {
                                  'model': self.prov_global_model}, self.subset_test_data)["accuracy"]
        logging.info(f"Sanity check: {acc}")
        # _ = input("Press Enter to continue")
        assert int(acc) == 0, "Sanity check failed"
        
        return acc

    def _setParticipatingClientsLabels(self) -> None:
        """
        Gathers and sets the union of all labels from the models of participating clients. This method
        iterates over each client in `client2model` and retrieves their corresponding labels from the
        `ALLROUNDSCLIENTS2CLASS` dictionary, aggregating all unique labels.

        This aggregated set of labels is crucial for identifying which classes are represented by the clients
        in the current round of the federated learning process and is used in subsequent operations
        to tailor data selection and model evaluations.

        Updates:
        - `participating_clients_labels`: A list of unique labels that represent the union of all classes
        associated with the current set of clients.

        Postconditions:
        - The `participating_clients_labels` attribute is updated with a list of all unique labels across clients.
        - Debug logging provides visibility into the derived labels for verification and troubleshooting.
        """
        labels = set()
        for c in self.client2model.keys():
            labels = labels.union(self.ALLROUNDSCLIENTS2CLASS[c])

        self.participating_clients_labels = list(labels)
        logging.debug(
            f"participating_clients_labels: {self.participating_clients_labels}"
        )

    def _eval_and_extract_wrong_preds(self, test_data):
        d = test_neural_network(self.train_cfg.model.arch, {
                                'model': self.prov_global_model}, test_data)  # type: ignore
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug(f"Accuracy on test data: {self.acc}")

        return d["eval_incorrect_indices"], d["eval_actual_labels"], d["eval_predicted_labels"]
    
    def _selectProvenanceData(self, central_test_data, min_per_label: int = 10) -> None:
        """
        Selects a subset of test data based on correct prediction indices to ensure the data used for 
        provenance analysis meets the required quality and distribution criteria. This method loads the 
        test data, identifies the correct predictions, and then creates a subset of the test data based 
        on these indices.

        Steps:
        - Loads the test data from the central server based on the distribution specified in `train_cfg`.
        - Uses `_getBalancedCorrectIndices` to identify indices of correctly predicted samples.
        - Creates a subset of the test data using these indices to form `subset_test_data`.

        Side Effects:
        - Updates `self.subset_test_data` with the subset of test data that contains correctly predicted samples.
        - Logs a message if no correct predictions are found, which is important for debugging and understanding
        the performance of the model under current configuration settings.

        Example Usage:
        - instance._selectProvenanceData()  # Potentially updates `self.subset_test_data`
        """

        label2flip = self.train_cfg.label2flip
        assert len(self.train_cfg.faulty_clients_ids) > 0, "No faulty clients"
        all_wrong_i, dataset_lablels, predicted_labels = self._eval_and_extract_wrong_preds(central_test_data)


        logging.info(f"Total wrong predictions: {len(all_wrong_i)}")
        logging.info(f'label2flip {label2flip}')

        # print first 5 elements in each list
        logging.info(f'Actual labels : {dataset_lablels}')
        logging.info(f'Predicted labels : {predicted_labels}')


        selected_wrong_indices = []

        for index_i in all_wrong_i:
            predicted_label = int(predicted_labels[index_i]) 
            true_label =  int(dataset_lablels[index_i])           
            # print(f'type of true_label {type(true_label)}, type of predicted_label {type(predicted_label)}')

            # print('label2flip.keys() ', label2flip.keys())
            # print('label2flip.values() ', label2flip.values())

            if predicted_label in  [int(l) for l in label2flip.values()]  and true_label in [int(l) for l in label2flip.keys()]:# and true_label in label2flip.keys():
                selected_wrong_indices.append(index_i)


        selected_wrong_indices  = selected_wrong_indices[:min_per_label]    
        self.subset_test_data = central_test_data.select(selected_wrong_indices)

        logging.info(f'Selected Actual labels : {dataset_lablels[selected_wrong_indices]}')
        logging.info(f'Selected Predicted labels : {predicted_labels[selected_wrong_indices]}')

        # _ = input("Press any key to continue")

        if len(self.subset_test_data) == 0:
            logging.info("No correct predictions found")


    def _computeEvalMetrics(self, input2prov) :
        """
        Computes evaluation metrics, particularly the accuracy of provenance data, by matching predicted clients
        (traced clients) against the actual data labels. The method calculates how often the traced client was
        responsible for the correct label according to the model's prediction.

        Parameters:
        - input2prov (list of dicts): A list of dictionaries where each dictionary contains mapping
                                    of traced clients and their confidence levels for specific inputs.

        Returns:
        - dict: A dictionary containing the calculated accuracy, formatted as:
            {'accuracy': float}, where accuracy is the proportion of correctly traced predictions.

        Details:
        - Creates a DataLoader for iterating over `subset_test_data`.
        - Extracts labels from the data loader.
        - Constructs a mapping of clients to classes they are responsible for (`client2class`).
        - Iterates over input to provenance mappings and checks if traced clients are responsible for
        the actual labels of the inputs.
        - Accumulates counts of correct tracings and the cumulative confidence level.
        - Calculates the accuracy as the ratio of correct tracings to the total number of inputs.

        Side Effects:
        - Logs detailed debugging information about the matching process and intermediate results,
        aiding in the transparency and traceability of the evaluation process.

        Example Usage:
        - eval_metrics = instance._computeEvalMetrics(input_provenance_data)
        """
        data_loader = torch.utils.data.DataLoader(self.subset_test_data, batch_size=1)
        
        target_labels = [data['label'].item() for data in data_loader]

        client2class = {c: self.ALLROUNDSCLIENTS2CLASS[c] for c in self.client2model}

        logging.debug(f"client2class: {client2class}")

        correct_tracing = 0

        true_labels = []
        predicted_labels = []

        for idx, prov_r in enumerate(input2prov):
            traced_client = prov_r["traced_client"]
            client2prov = prov_r["client2prov"]

            target_l = target_labels[idx]
            responsible_clients = [f"{c}" for c in self.train_cfg.faulty_clients_ids]

            res_c_string = ','.join(map(str, responsible_clients))

            logging.info(
                f'            *********** Input Label: {target_l}, Responsible Client(s): {res_c_string}  *************')

            if traced_client in responsible_clients:
                logging.info(
                    f"     Traced Client: c{traced_client} || Prediction = True")
                correct_tracing += 1
                predicted_labels.append(1)
                true_labels.append(1)
            else:
                logging.info(

                    f"     Traced Client: c{traced_client} || Prediction = False")
                predicted_labels.append(0)
                true_labels.append(1)

            c2nk_batches = {
                f'c{c}': self.client2num_examples[c] for c in client2prov.keys()}
            c2nk_total = {f'c{c}': sum(
                client2class[c].values()) for c in client2prov.keys()}  # type: ignore
            c2nk_label = {f'c{c}': client2class[c].get(  # type: ignore
                target_l, 0) for c in client2prov.keys()}
            c2nk_label = {c: v for c, v in c2nk_label.items() if v > 0}

            client2prov_score = {f'c{c}': round(
                p, 2) for c, p in client2prov.items()}
            logging.info(f"    Client Prov Score:     {client2prov_score}")
            logging.info(f"    C2nk_label_{target_l}:      {c2nk_label}")
            logging.info(f"    C2nk_batches     :     {c2nk_batches}")
            logging.info(f"    C2nk_total       :     {c2nk_total}")

            c_label = max(c2nk_label, key=c2nk_label.get)  # type: ignore
            c_batch = max(c2nk_batches, key=c2nk_batches.get)  # type: ignore
            logging.info(
                f'    {c_label} has most label-{target_l} = {c2nk_label[c_label]} || {c_batch} has max data = {c2nk_batches[c_batch]}')
            logging.info('\n')

        eval_metrics = get_prov_eval_metrics(true_labels, predicted_labels)

        print(f'eval metrics {eval_metrics}')

        a = correct_tracing / len(input2prov)
        assert a == eval_metrics['Accuracy'], "Accuracy mismatch"
        return eval_metrics

    def run(self): 
        r = self._sanityCheck()
        if r is None:
            prov_result = {'Error': "No data found for provenance analysis"}
            return prov_result
            

        sart_time = time.time()
        nprov = NeuronProvenance(cfg=self.prov_cfg, arch=self.train_cfg.model.arch, test_data=self.subset_test_data, gmodel=self.prov_global_model,  # type: ignore
                                 c2model=self.client2model, num_classes=self.train_cfg.dataset.num_classes, c2nk=self.client2num_examples)  # type: ignore
        input2prov = nprov.computeInputProvenance()
        eval_metrics = self._computeEvalMetrics(input2prov)
        end_time = time.time()

        logging.info(f"[R {self.round_id}] Provenance Accuracy %:= {eval_metrics['Accuracy']} || Total Inputs Used In Prov: {len(self.subset_test_data)} || GM_(loss, acc) ({self.loss},{self.acc})")
        # wandb.log(
        #     {"prov_accuracy": eval_metrics['accuracy'], 'gm_loss': self.loss, 'gm_acc': self.acc})


        fed_debug_result = {}

        if self.prov_cfg.model.arch == 'cnn':
            temp_dict =  round_lambda_fed_debug_func(self.prov_cfg, self.round_key)
            fed_debug_result['FedDebug Accuracy'] = temp_dict['eval_metrics']['accuracy']
            fed_debug_result['FedDebug avg_fault_localization_time'] = temp_dict['avg_fault_localization_time']
            fed_debug_result['FedDebug avg_input_gen_time'] = temp_dict['avg_input_gen_time']
            print(f'fed_debug_result --> {fed_debug_result}')


        # print(f'----------------> total subset_test_data {len(self.subset_test_data)}')
        prov_result = {
            "clients": list(self.client2model.keys()),
            "data_points": len(self.subset_test_data),
            "eval_metrics": eval_metrics,
            "test_data_acc": self.acc,
            "test_data_loss": self.loss,
            "avg_prov_time_per_input": (end_time - sart_time)/ len(self.subset_test_data),
            "round_id": self.round_id,
            'prov_layers': self.save_prov_layers,
        }

        prov_result = {**prov_result, **fed_debug_result}
        print(f'Prov result {prov_result}')
        return prov_result


def _get_round_keys(fl_key, train_cache_path):
    training_cache = Index(train_cache_path)

    r_keys = []
    for k in training_cache.keys():
        if fl_key == k:
            continue
        elif fl_key in k and len(k) > len(fl_key):
            r_keys.append(k)

    if len(r_keys) == 0:
        raise ValueError(
            f"No rounds found for key {fl_key}. Please check the training cache.")

    
    return r_keys


def _checkAlredyDone(fl_config_key: str, results_cache):
    if fl_config_key in results_cache:
        d = results_cache[fl_config_key]
        return d["round2prov_result"]
    return []


def _roundLambdaProv(cfg, round_key, central_test_data):
    if len(cfg.faulty_clients_ids) > 0:
        round_prov = FederatedProvFalse(
            cfg, round_key, central_test_data, t=None)
    else:
        round_prov = FederatedProvTrue(
            cfg, round_key, central_test_data, t=None)
    try:
        prov_result_dict = round_prov.run()
    except Exception as e:
        logging.error(
            f"Error in running provenance for round {round_key}. Error: {e}")
        prov_result_dict = {'Error': e}

    return prov_result_dict


def _run_and_save_prov_result_in_cache(cfg):
    round2prov_result = []
    train_cache_path = cfg.storage.dir + cfg.storage.train_cache_name
    prov_results_cache = Index(
        cfg.storage.dir + cfg.storage.results_cache_name)

    if cfg.check_prov_cache:
        round2prov_result = _checkAlredyDone(cfg.exp_key, prov_results_cache)
        # print(f"round2prov_result {round2prov_result}")
        if len(round2prov_result) > 0:
            logging.info(
                f">> Done.Provenance of key {cfg.exp_key} is already done.")
            return round2prov_result

    logging.info(f"Starting provenance analysis for {cfg.exp_key}...")
    rounds_keys = _get_round_keys(cfg.exp_key, train_cache_path)

    central_test_data = get_clients_server_data(cfg)['server_testdata']

    # total test data size
    logging.info(f"Total test data size: {len(central_test_data)}")

    

    start_time = time.time()

    logging.info(f'Number of parallel processes: {cfg.parallel_processes}')

    if cfg.parallel_processes > 1:
        with Pool(processes=cfg.parallel_processes) as p:
            logging.info(
                f"Running provenance analysis for {len(rounds_keys)} rounds in parallel...")
            round2prov_result = p.starmap(_roundLambdaProv, [(
                cfg, round_key, central_test_data) for round_key in rounds_keys])
            p.close()
            p.join()
        
        

    else:
        logging.info(
            f"Running provenance analysis for {len(rounds_keys)} rounds sequentially...")
        round2prov_result = [_roundLambdaProv(
            cfg, round_key, central_test_data) for round_key in rounds_keys]

    end_time = time.time()
    avg_prov_time_per_round = -1

    if len(rounds_keys) > 0:
        avg_prov_time_per_round = (end_time - start_time) / len(rounds_keys)

    prov_results_cache[cfg.exp_key] = {
        "round2prov_result": round2prov_result, "prov_cfg": cfg, "training_cache_path": train_cache_path, "avg_prov_time_per_round": avg_prov_time_per_round}

    logging.info(
        f"Provenance results saved for {cfg.exp_key}, avg provenance time per round: {avg_prov_time_per_round} seconds")


def _get_all_train_cfgs_from_train_cache(cache):
    cfgs = []

    filter_keys = [key for key in cache.keys() if key.find("-round:") == -1]

    for key in filter_keys:
        if key.find('Temp--') != -1:
            continue
        logging.info(f"key: {key}")

        if "train_cfg" in cache[key] and cache[key]['complete']:
            cfgs.append(cache[key]["train_cfg"])
    return cfgs


def run_full_cache_provenance(cfg):
    all_train_cfgs = _get_all_train_cfgs_from_train_cache(
        Index(cfg.storage.dir + cfg.storage.train_cache_name))
    for train_cfg in all_train_cfgs:
        train_cfg.parallel_processes = cfg.parallel_processes
        train_cfg.check_prov_cache = cfg.check_prov_cache
        try:
            _run_and_save_prov_result_in_cache(train_cfg)
        except Exception as e:
            logging.error(f"Error: {e}")
            logging.error(
                f"Error in provenance experiment: {train_cfg.exp_key}")


def given_key_provenance(cfg):
    _run_and_save_prov_result_in_cache(cfg)
