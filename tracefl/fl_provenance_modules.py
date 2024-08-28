from tracefl.models import test_neural_network, initialize_model
from tracefl.neuron_provenance import NeuronProvenance, getAllLayers
from tracefl.utils import get_prov_eval_metrics
from typing import Dict, List, Optional, Tuple
import torch
import logging
from diskcache import Index
import time

class FederatedProvTrue:
    def __init__(self, prov_cfg, round_key: str, server_test_data, t) -> None:
        self.t = t
        self.prov_cfg = prov_cfg
        self.round_key = round_key
        self._extractRoundID()
        self._loadTrainingConfig()
        self._initializeAndLoadModels()
        self._setParticipatingClientsLabels()
        self._selectProvenanceData(server_test_data)

    def _modelInitializeWrapper(self):
        m = initialize_model(self.train_cfg.model.name,
                             self.train_cfg.dataset)  # type: ignore
        return m['model']

    def _extractRoundID(self) -> None:
        self.round_id = self.round_key.split(":")[-1]

    def _loadTrainingConfig(self) -> None:
        """
        Loads the training configuration from a cached storage defined in the provenance configuration (`prov_cfg`). 
        This method initializes the `training_cache` with the directory and file name specified in the config,
        retrieves the experiment dictionary using the configured experiment key (`exp_key`), and sets up
        various class attributes needed for further operations.

        Specifically, this method sets:
        - `train_cfg`: The training configuration which includes model details, dataset information, etc.
        - `ALLROUNDSCLIENTS2CLASS`: A dictionary mapping from client identifiers to the classes they are responsible for.

        Assumes that the necessary directories and cache names are correctly specified in `prov_cfg` and that
        `prov_cfg.exp_key` correctly indexes the required data within the cache.

        Raises:
            KeyError: If `prov_cfg.exp_key` does not exist in the training cache.
        """

        self.training_cache = Index(
            self.prov_cfg.storage.dir + self.prov_cfg.storage.train_cache_name)
        exp_dict = self.training_cache[self.prov_cfg.exp_key]
        self.train_cfg = exp_dict["train_cfg"]  # type: ignore
        self.ALLROUNDSCLIENTS2CLASS = exp_dict["client2class"]  # type: ignore

    def _initializeAndLoadModels(self) -> None:
        """
        Initializes and loads the global model and client-specific models for the current round of training.
        This method retrieves the model weights from the training cache using the current round key and
        sets up both the global model and the individual models for each participating client.

        Steps:
        1. Retrieve the workspace dictionary (`round2ws`) from the training cache using `round_key`.
        2. Initialize and load the global model:
        - The global model is initialized with the model name and dataset configuration from `train_cfg`.
        - The model's state is set with weights (`gm_ws`) retrieved from `round2ws`.
        - The model is then moved to CPU and set to evaluation mode.
        3. Initialize and load models for each client:
        - For each client, a model is initialized and loaded similarly to the global model.
        - Client models are also moved to CPU and set to evaluation mode.
        - Each client model is stored in `client2model` dictionary mapping client IDs to their respective models.

        Attributes updated:
        - `prov_global_model`: The global model configured for the current round.
        - `client2model`: A dictionary mapping client IDs to their respective initialized and loaded models.

        Raises:
            KeyError: If the necessary keys ('gm_ws' or 'client2ws') are missing in `round2ws`.
        """
        logging.info(
            f'\n\n             ----------Round key  {self.round_key} -------------- \n')
        round2ws = self.training_cache[self.round_key]
        self.client2num_examples = round2ws["client2num_examples"]
        self.prov_global_model = self._modelInitializeWrapper()
        self.prov_global_model.load_state_dict(
            round2ws["gm_ws"])  # type: ignore
        self.prov_global_model = self.prov_global_model.cpu().eval()  # type: ignore

        self.save_prov_layers = set(
            [type(l) for l in getAllLayers(self.prov_global_model)])
        logging.debug(f"Layers used in Provenance: {self.save_prov_layers}")

        self.client2model = {}
        for cid, ws in round2ws["client2ws"].items():  # type: ignore
            cmodel = self._modelInitializeWrapper()
            cmodel.load_state_dict(ws)  # type: ignore
            cmodel = cmodel.cpu().eval()  # type: ignore
            self.client2model[cid] = cmodel

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

    def _evalAndExtractCorrectPredsTransformer(self, test_data):

        d = test_neural_network(self.train_cfg.model.arch, {
                                'model': self.prov_global_model}, test_data)  # type: ignore
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug(f"Accuracy on test data: {self.acc}")
        return d["eval_correct_indices"], d["eval_actual_labels"]

    def _balanceDatasetByLabel(self, correct_indices: torch.Tensor, dataset_labels: torch.Tensor, min_per_label: int) -> torch.Tensor:
        """
        Balances the dataset by ensuring that each label has at least a specified number of data points,
        selected from the indices of correctly predicted labels. This method is crucial for preparing
        a fair and balanced subset for further analysis or testing, particularly in scenarios where class
        distribution is uneven.

        Parameters:
        - correct_indices (torch.Tensor): Tensor of indices corresponding to correctly predicted samples.
        - dataset_labels (torch.Tensor): Tensor of labels for the dataset, which must correspond in size and order
                                        to `correct_indices`.
        - min_per_label (int): The minimum number of samples required for each label in the resulting subset.

        Returns:
        - torch.Tensor: A tensor of indices representing a balanced subset of the dataset, where each label
                        represented in `participating_clients_labels` has at least `min_per_label` entries, if possible.

        Raises:
        - ValueError: If `min_per_label` is greater than the available correct predictions for a required label,
                    which could prevent forming a fully balanced subset as specified.

        Detailed Steps:
        - Initializes a dictionary to count occurrences of each label (`label2count`).
        - Iterates over each label present in the `participating_clients_labels`.
        - Selects and counts the occurrences of each label among the correctly predicted indices.
        - Accumulates the first `min_per_label` indices for each label to ensure balanced representation.
        - Concatenates these indices into a single tensor to form the balanced dataset subset.

        Example Usage:
        - balanced_indices = instance._balanceDatasetByLabel(correct_indices, dataset_labels, 3)
        """
        balanced_indices = []
        logging.debug(
            f'participating_clients_labels {self.participating_clients_labels}')
        for l in self.participating_clients_labels:
            # print(f'Label {l}')
            selected_labels = dataset_labels[correct_indices]
            # print(f'selected_labels { set(selected_labels)}')
            temp_bools = selected_labels == l
            # print(f'temp_bools {temp_bools}')
            temp_correct_indxs = correct_indices[temp_bools]
            if len(temp_correct_indxs) >= min_per_label:
                balanced_indices.append(temp_correct_indxs[:min_per_label])
        if len(balanced_indices) > 0:
            balanced_indices = torch.cat(balanced_indices)

        return balanced_indices

    # def _balanceDatasetByLabel2(self,  ds,  min_per_label: int):
    #     all_ds = []
    #     for l in self.participating_clients_labels:
    #         ds_of_l = ds.filter(lambda x: x['label'] == l)
    #         # ds_of_l = Dataset.from_dict(ds_of_l[:len(ds_of_l)])
    #         # only min_per_label
    #         if len(ds_of_l) > min_per_label:
    #             ds_of_l = ds_of_l.take(min_per_label)

    #         if len(ds_of_l) > 0:
    #             all_ds.append(ds_of_l)
    #             # print(ds_of_l)

    #     balanced_ds = concatenate_datasets(all_ds)

    #     return balanced_ds

    def _selectProvenanceData(self, central_test_data, min_per_label: int = 2) -> None:
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
        all_correct_i, dataset_lablels = self._evalAndExtractCorrectPredsTransformer(
            central_test_data)

        # correct_ds = Dataset.from_dict(central_test_data[all_correct_i])

        balanced_indices = self._balanceDatasetByLabel(
            all_correct_i, dataset_lablels, min_per_label)  # type: ignore

        self.subset_test_data = central_test_data.select(balanced_indices)

        if len(self.subset_test_data) == 0:
            logging.info("No correct predictions found")

    def _sanityCheck(self):

        if len(self.subset_test_data) == 0:
            raise ValueError("No correct predictions found")

        acc = test_neural_network(self.train_cfg.model.arch, {
                                  'model': self.prov_global_model}, self.subset_test_data)["accuracy"]
        logging.info(f"Sanity check: {acc}")
        # _ = input("Press Enter to continue")
        assert int(acc) == 1, "Sanity check failed"
        
        return acc

    def _computeEvalMetrics(self, input2prov: List[Dict]) -> Dict[str, float]:
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
        data_loader = torch.utils.data.DataLoader(  # type: ignore
            self.subset_test_data, batch_size=1)
        target_labels = [data['label'].item() for data in data_loader]

        client2class = {
            c: self.ALLROUNDSCLIENTS2CLASS[c] for c in self.client2model}

        logging.debug(f"client2class: {client2class}")

        correct_tracing = 0

        true_labels = []
        predicted_labels = []

        for idx, prov_r in enumerate(input2prov):
            traced_client = prov_r["traced_client"]
            client2prov = prov_r["client2prov"]

            target_l = target_labels[idx]
            responsible_clients = [
                cid for cid, c_labels in client2class.items() if target_l in c_labels]

            res_c_string = ','.join(map(str, responsible_clients))

            logging.info(
                f'            *********** Input Label: {target_l}, Responsible Client(s): {res_c_string}  *************')

            if target_l in client2class[traced_client]:
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

    def run(self) -> Dict[str, any]:  # type: ignore
        r = self._sanityCheck()
        if r is None:
            prov_result = {
                "clients": list(self.client2model.keys()),
                "data_points": len(self.subset_test_data),
                "eval_metrics": {},
                "test_data_acc": self.acc,
                "test_data_loss": self.loss,
                "prov_time": -1,
                "round_id": self.round_id,
                'prov_layers': self.save_prov_layers,
            }

        sart_time = time.time()
        nprov = NeuronProvenance(cfg=self.prov_cfg, arch=self.train_cfg.model.arch, test_data=self.subset_test_data, gmodel=self.prov_global_model,  # type: ignore
                                 c2model=self.client2model, num_classes=self.train_cfg.dataset.num_classes, c2nk=self.client2num_examples)  # type: ignore
        input2prov = nprov.computeInputProvenance()
        eval_metrics = self._computeEvalMetrics(input2prov)
        end_time = time.time()

        logging.info(f"[R {self.round_id}] Provenance Accuracy %:= {eval_metrics['Accuracy']} || Total Inputs Used In Prov: {len(self.subset_test_data)} || GM_(loss, acc) ({self.loss},{self.acc})")
        # wandb.log(
        #     {"prov_accuracy": eval_metrics['accuracy'], 'gm_loss': self.loss, 'gm_acc': self.acc})

        prov_result = {
            "clients": list(self.client2model.keys()),
            "data_points": len(self.subset_test_data),
            "eval_metrics": eval_metrics,
            "test_data_acc": self.acc,
            "test_data_loss": self.loss,
            "prov_time": end_time - sart_time,
            "round_id": self.round_id,
            'prov_layers': self.save_prov_layers,
        }

        return prov_result

