import logging
import torch
import torch.nn.functional as F
import transformers
import transformers.models.bert.modeling_bert as modeling_bert
from transformers import AutoModelForSequenceClassification,   AutoTokenizer
import transformers.models.openai.modeling_openai as modeling_openai
from tracefl.utils import compute_importance
import transformers.models.roberta.modeling_roberta as modeling_roberta

from tracefl.models import test_neural_network




# def min_max_normalize(tensor):
#     """
#     Normalize a tensor using min-max normalization.

#     Parameters:
#     tensor (torch.Tensor): The input tensor to be normalized.

#     Returns:
#     torch.Tensor: The min-max normalized tensor.
#     """
#     min_val = tensor.min()
#     max_val = tensor.max()
#     normalized_tensor = (tensor - min_val) / (max_val - min_val)
#     return normalized_tensor


class NeuronProvenance:
    def __init__(self, cfg, arch, test_data, gmodel, c2model, num_classes, c2nk):
        """
        Initializes a new instance of the NeuronProvenance class, which is designed to analyze and determine the contributions of different clients' data to predictions of a federated learning global model. 

        Parameters:
            test_data (List[torch.Tensor]): The dataset used for testing the global model. 
            gmodel (torch.nn.Module): The global model that will be evaluated. This model should be a aggregating instance of c2model using techniques like FedAvg or FedProx in federated learning.
            c2model (Dict[int, torch.nn.Module]): A dictionary mapping client IDs to their respective models. These clients' models are trained in FL training round.
            num_classes (int): The number of classes that the model can classify. This is used primarily for handling the output layer's characteristics in some methods.
            device (torch.device): The device (e.g., CPU or GPU) on which the model computations are to be performed.
        This constructor also initializes a list of client IDs extracted from the keys of the `c2model` dictionary, which is used throughout various methods in the class to iterate over client-specific information.
        """
        self.arch = arch
        self.cfg = cfg
        self.test_data = test_data
        self.gmodel = gmodel
        self.c2model = c2model
        self.num_classes = num_classes
        self.device = self.cfg.device
        self.c2nk = c2nk  # client2num_examples
        self.client_ids = list(self.c2model.keys())
        self.layer_importance = compute_importance(len(getAllLayers(gmodel)))
        logging.info(f'client ids: {self.client_ids}')
        self.pk = {
            cid: self.c2nk[cid] / sum(self.c2nk.values()) for cid in self.c2nk.keys()}
        if self.cfg.client_weights_normalization:
            logging.debug('>> clients weights are normaized')
            self._inplaceScaleClientWs()

    def _checkAnomlies(self, t):
        inf_mask = torch.isinf(t)
        nan_mask = torch.isnan(t)
        if inf_mask.any() or nan_mask.any():
            logging.error(f"Inf values: {torch.sum(inf_mask)}")
            logging.error(f"NaN values: {torch.sum(nan_mask)}")
            logging.error(f"Total values: {torch.numel(t)}")
            # logging.error(f"Total values: {t}")
            raise ValueError("Anomalies detected in tensor")
    # def _calculateLayerContribution2(self, global_neurons_outputs: torch.Tensor, global_layer_grads: torch.Tensor, client2outputs: Dict[int, torch.Tensor]) -> Dict[int, float]:
    #     client2avg = {cid: 0.0 for cid in self.client_ids}
    #     self._checkAnomlies(global_neurons_outputs)
    #     self._checkAnomlies(global_layer_grads)
    #     global_layer_grads = global_layer_grads.flatten()

    #     vector_len  = len(global_layer_grads)
    #     for cli in self.client_ids:
    #         cli_acts = client2outputs[cli].to(self.device).flatten()

    #         self._checkAnomlies(cli_acts)
    #         # cli_acts = F.softmax(cli_acts, dim=0) 
        
    #         cli_part =  torch.dot(cli_acts, global_layer_grads)
    #         client2avg[cli] = cli_part.item() / vector_len
    #         logging.debug(f"Client {cli} contribution: {client2avg[cli]}")

    def _calculateLayerContribution(self, global_neurons_outputs: torch.Tensor, global_layer_grads: torch.Tensor, client2outputs, layer_id: int):
        # _ = input("Press Enter to continue...")
        client2avg = {cid: 0.0 for cid in self.client_ids}
        self._checkAnomlies(global_neurons_outputs)
        self._checkAnomlies(global_layer_grads)
        global_layer_grads = global_layer_grads.flatten()

        # global_neurons_outputs_with_grad = global_neurons_outputs.flatten() * global_layer_grads        
        # imp_neurons_bools  = global_neurons_outputs_with_grad != 0
        # vector_len  = torch.sum(imp_neurons_bools).item()
        # if vector_len == 0:
        #     logging.warning(f"No important neurons detected. Check this key {self.cfg.exp_key}")
        #     # return client2avg
        #     vector_len = 100
        # select neurons with non-zero output


        
        for cli in self.client_ids:
            cli_acts = client2outputs[cli].to(self.device).flatten()
            self._checkAnomlies(cli_acts)
            # cli_part =  torch.dot(cli_acts[imp_neurons_bools], global_layer_grads[imp_neurons_bools])
            cli_part =  torch.dot(cli_acts, global_layer_grads)           
            
            client2avg[cli] = cli_part.item() * self.layer_importance[layer_id]
            # logging.info(f"Client {cli} contribution: {client2avg[cli]}")
            cli_acts = cli_acts.cpu()

        # cli with max contribution
        max_contributor = max(client2avg, key=client2avg.get) # type: ignore
        logging.debug(f"Max contributor: {max_contributor}")
        return client2avg

    def _mapClientLayerContributions(self, layer_id: int):
        """
        Computes the contribution of each client's to the output of a specific layer for each input in the test dataset. This method evaluates client-specific layers against the global model's neuron inputs and outputs, determining the influence of each client's on the global model (self.gmodel).

        Parameters:
            layer_id (int): The index of the layer in the global model for which contributions are being calculated.

        Returns:
            Dict[int, Dict[int, float]]: A dictionary where each key is an input ID and each value is another dictionary mapping client IDs to their contribution scores for this particular layer.
        """

        # print(f"Mapping client contributions for layer {layer_id}")

        client2layers = {cid: getAllLayers(cm)
                         for cid, cm in self.c2model.items()}
        global_neurons_inputs = self.global_neurons_inputs_outputs_batch[layer_id][0]
        global_neurons_inputs = global_neurons_inputs.to(self.device)

        global_neurons_outputs = self.global_neurons_inputs_outputs_batch[layer_id][1]
        # print(global_neurons_outputs)

        # check if global_neurons_outputs is a tuple
        if isinstance(global_neurons_outputs, tuple) or isinstance(global_neurons_outputs, list):
            # print(f"Tuple detected of length {len(global_neurons_outputs)}")
            # print(f"2nd element of tuple: {global_neurons_outputs[1]}")
            assert len(
                global_neurons_outputs) == 1, f"Expected 1 element in tuple, got {len(global_neurons_outputs)}"
            global_neurons_outputs = global_neurons_outputs[0]
        # else:
        #     print(f"Tuple not detected. Type is {type(global_neurons_outputs)} detected")
        #     raise ValueError("Tuple not detected")
        # print(f"Global neurons outputs shape: {global_neurons_outputs}")
        global_neurons_outputs = global_neurons_outputs.to(self.device)

        # global_grads =  self.inputs2layer_grads[layer_id][1].to(self.device)

        c2l = {cid: client2layers[cid][layer_id] for cid in self.client_ids}
        clinet2outputs = {c: self._evaluateLayer(
            l, global_neurons_inputs) for c, l in c2l.items()}

        input2client2contribution = {}
        for input_id in range(len(self.test_data)):  # for per input in the batch
            logging.debug(
                f"Mapping client contributions for input {input_id} for layer {layer_id}")
            c2out_per_input = {
                cid: clinet2outputs[cid][input_id] for cid in self.client_ids}

            glayer_grads = torch.squeeze(
                self.inputs2layer_grads[input_id][layer_id][1]).to(self.device)

            c2contribution = self._calculateLayerContribution(
                global_neurons_outputs=global_neurons_outputs[input_id], global_layer_grads=glayer_grads, client2outputs=c2out_per_input, layer_id=layer_id)
            input2client2contribution[input_id] = c2contribution

        return input2client2contribution

    def _inplaceScaleClientWs(self):
        """
        Scales the client weights based on the number of data points each client has. This scaling is necessary to ensure that clients with more data points have a proportionally larger impact on the global model's predictions.

        Returns:
            Dict[int, torch.nn.Parameter]: A dictionary mapping client IDs to their scaled weights.
        """
        logging.debug(
            "Scaling client weights based on the number of data points each client has.")
        # for param in self.gmodel.parameters():
        #     print(param.data)
        #     param.data =  param.data * 0.0
        #     param.data =  param.data -1
        #     break

        # for param in self.gmodel.parameters():
        #     print(param.data)
        #     break
        logging.debug(f'Total clients in c2nk: {len(self.c2nk)}')
        logging.debug(f'Total clients in c2model: {len(self.c2model)}')

        ids1 = list(self.c2model.keys())
        ids2 = list(self.c2nk.keys())
        logging.debug(f'ids1: {ids1}')
        logging.debug(f'ids2: {ids2}')

        # temp_global  =  copy.deepcopy(list(self.c2model.values())[0])

        for cid in self.c2model.keys():
            scale_factor = self.c2nk[cid] / sum(self.c2nk.values())
            logging.debug(
                f"Scaling client {cid} by {scale_factor}, nk = {self.c2nk[cid]}")
            # input("Press Enter to continue...")
            with torch.no_grad():
                for cparam in self.c2model[cid].parameters():
                    cparam.data = scale_factor * cparam.data
                self.c2model[cid] = self.c2model[cid].eval().cpu()

        # with torch.no_grad():
        #     for idx, gparam in enumerate(temp_global.parameters()):
        #         gparam.data = gparam.data * 0.0
        #         for cid in self.c2model.keys():
        #             scale_factor = self.c2nk[cid] / sum(self.c2nk.values())
        #             for idx2, cparam in enumerate(self.c2model[cid].parameters()):
        #                 if idx == idx2:
        #                     cparam.data = scale_factor * cparam.data
        #                     gparam.data += cparam.data
        #                     break
        # # global model parameters
        # logging.debug("Global model parameters after scaling")
        # for param in self.gmodel.parameters():
        #     print(param.data)
        #     break

        # logging.debug("temp_global parameters after scaling")
        # for param in temp_global.parameters():
        #     print(param.data)
        #     break


        # return client2ws

    def _captureLayerIO(self):
        hook_manager = HookManager()
        glayers = getAllLayers(self.gmodel)
        # logging.debug(f"all layers in global model: {glayers}")
        logging.debug(f"Total layers in global model: {len(glayers)}")
        hooks_forward = [hook_manager.insertForwardHook(
            layer) for layer in glayers]
        self.gmodel.eval()
        self.gmodel = self.gmodel.to(self.device)
        test_neural_network(self.arch, {'model': self.gmodel}, self.test_data,
                        batch_size=len(self.test_data))
        hook_manager.removeHooks(hooks_forward)
        self.global_neurons_inputs_outputs_batch = hook_manager.forward_hooks_storage
        hook_manager.clearStorages()

    def _captureLayerGradients(self):
        self.inputs2layer_grads = []
        for m_input in torch.utils.data.DataLoader(self.test_data, batch_size=1): # type: ignore
            hook_manager = HookManager()
            setGradientsofModel(self.arch, self.gmodel,
                                m_input, self.device, hook_manager)
            self.inputs2layer_grads.append(hook_manager.backward_hooks_storage)
            hook_manager.clearStorages()

    def _evaluateLayer(self, client_layer: torch.nn.Module, global_neurons_inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluates a specific neural network layer using provided input tensors and transfers the outputs to the CPU. This function is particularly useful in scenarios where layer-specific processing is required independently of the full network operations, such as in detailed analysis or during debugging.

        Parameters:
            client_layer (torch.nn.Module): The neural network layer to be evaluated. This layer must be compatible with the provided input tensor dimensions.
            global_neurons_inputs (torch.Tensor): The input tensor to feed into the client_layer. The shape of this tensor must match the input requirements of the client_layer.

        Returns:
            torch.Tensor: The output from the client_layer after processing the input tensor. This output tensor is moved to the CPU.

        Example:
            >>> example_layer = nn.Linear(10, 5)  # Create a linear layer expecting inputs of size 10 and outputting size 5
            >>> example_input = torch.rand(1, 10)  # Generate a random tensor of shape [1, 10]
            >>> output_tensor = _evaluateLayer(example_layer, example_input)
            >>> print(output_tensor)  # Output tensor from the layer, now on CPU
        """

        client_layer = client_layer.eval().to(device=self.device)
        client_neurons_outputs = client_layer(global_neurons_inputs)

        if isinstance(client_neurons_outputs, tuple) or isinstance(client_neurons_outputs, list):
            # print(f"Tuple detected of length {len(client_neurons_outputs)}")
            client_neurons_outputs = client_neurons_outputs[0].cpu()
        else:
            client_neurons_outputs = client_neurons_outputs.cpu()

        client_layer = client_layer.cpu()
        return client_neurons_outputs

    def _aggregateClientContributions(self, input_id: int, layers2prov):
        """
        Calculates the total contributions of each client across all specified layers on the given input to a global model. This method aggregates the contribution scores for each client, providing a comprehensive view of each client's influence on the model's output for that input.

        Parameters:
            input_id (int): The index of the input in the test dataset for which contributions are being calculated.
            layers2prov (List[Dict[int, Dict[int, float]]]): A list of dictionaries for each layer, where each dictionary maps input IDs to another dictionary. This inner dictionary maps client IDs to their contribution scores for the respective layer.

        Returns:
            Dict[int, float]: A dictionary mapping each client ID to their total contribution score across all layers for the specified input. This score is the sum of contributions from all layers for the given client on a specific input.

        Detailed Steps:
        - Initializes a dictionary to store the total contributions for each client with all values set to zero.
        - Iterates through the contribution data for each layer specific to the provided input.
        - Sums up the contributions for each client across all the layers to calculate the total contribution for each client.
        """
        client2totalcont = {c: 0.0 for c in self.client_ids}
        clients_prov = [i2prov[input_id] for i2prov in layers2prov]
        for c2lsum in clients_prov:
            for cid in self.client_ids:
                client2totalcont[cid] += c2lsum[cid]
        return client2totalcont

    def _normalizeContributions(self, contributions):
        """
        Normalizes the contribution scores using the softmax function to ensure they sum to one, making them directly comparable. The normalized scores are then sorted in descending order based on the contribution magnitude.

        Parameters:
            contributions (Dict[int, float]): A dictionary mapping client IDs to their raw contribution scores.

        Returns:
            Dict[int, float]: A dictionary mapping client IDs to their normalized contribution scores, sorted from highest to lowest contribution.

        Explanation:
        - The softmax function is applied to the raw contribution values to convert them into a probability distribution.
        - Each client's normalized score is then mapped back to the client ID.
        - The resulting dictionary is sorted by contribution scores in descending order to easily identify the most influential clients.
        """
        conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
        client2prov = {cid: v.item() for cid, v in zip(self.client_ids, conts)}
        return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))

    def _aggregateInputContributions(self, layers2prov):
        """
        Aggregates and normalizes the contributions of each client across multiple layers for each input to the global model in the test dataset, identifying the most influential client per input.

        Parameters:
            layers2prov (List[Dict[int, Dict[int, float]]]): A list where each element is a dictionary mapping input IDs to another dictionary. This inner dictionary maps client IDs to their contribution scores for a specific layer.

        Returns:
            List[Dict[str, Union[int, Dict[int, float]]]]: A list of dictionaries, where each dictionary contains the ID of the most influential client ('traced_client') and another dictionary mapping client IDs to their normalized contribution scores ('client2prov') for each input.

        Detailed Steps:
        - For each input in the test dataset:
            - Calculate total contributions of each client by summing across layers.
            - Normalize these total contributions to compare them on a consistent scale.
            - Determine the client with the maximum contribution for each input.
        """

        input2prov = []
        for input_id in range(len(self.test_data)):
            client_conts = self._aggregateClientContributions(
                input_id, layers2prov)
            normalized_conts = self._normalizeContributions(client_conts)
            traced_client = max(
                normalized_conts, key=normalized_conts.get)  # type: ignore
            input2prov.append({
                "traced_client": traced_client,
                "client2prov": normalized_conts
            })
        return input2prov

    def computeInputProvenance(self) :
        """
        Orchestrates the computation of neurons' inputs, outputs, and gradients across all layers of the global model, then calculates and returns the provenance statistics for each input in the test dataset. This method identifies which client's model contributed most significantly to the Federated Learning global model's predictions.

        Returns:
            List[Dict[str, Union[int, Dict[int, float]]]]: A list where each item is a dictionary containing the client ID that had the maximum influence ('traced_client') and a dictionary of all clients with their respective normalized contributions ('client2prov').

        Detailed Workflow:
        - Computes the neurons' inputs and outputs across of global model all layers using forward hooks.
        - Computes the gradients for these neurons using backward hooks.
        - Aggregates contributions from each client per layer in the global model and then across all inputs.
        - Normalizes and determines the most influential client whose weights contributed most to the global model prediction for each input based on these contributions.
        """

        self._captureLayerIO()
        self._captureLayerGradients()

        layers2prov = []
        for layer_id in range(len(self.global_neurons_inputs_outputs_batch)):
            client2cont = self._mapClientLayerContributions(layer_id)
            layers2prov.append(client2cont)

        input2prov = self._aggregateInputContributions(layers2prov)
        return input2prov


class HookManager:
    def __init__(self):
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def insertForwardHook(self, layer):
        def forward_hook(module, input_tensor, output_tensor):
            # assert len(
            #     input_tensor) == 1, f"Expected 1 input, got {len(input_tensor)}"

            try:
                # Handle the input as a tuple, get the first element
                input_tensor = input_tensor[0]
                input_tensor = input_tensor.detach()
            except Exception as e:
                # logging.debug(f"Error processing input in forward hook: {e}")
                pass

            input_tensor = input_tensor.detach()
            output_tensor = output_tensor
            self.forward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_forward_hook(forward_hook)
        return hook

    def insertBackwardHook(self, layer):
        def backward_hook(module, input_tensor, output_tensor):
            # assert len(
            #     input_tensor) == 1, f"Expected 1 input, got {len(input_tensor)}"
            try:
                input_tensor = input_tensor[0]
                output_tensor = output_tensor[0]
                input_tensor = input_tensor.detach()
                output_tensor = output_tensor.detach()

            except Exception as e:
                # logging.debug(f"Error processing input in backward hook: {e}")
                pass
            try:
                input_tensor = input_tensor.detach()
            except Exception as e:
                pass
            try:
                output_tensor = output_tensor.detach()
            except Exception as e:
                pass

            self.backward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_full_backward_hook(backward_hook)
        return hook

    def clearStorages(self):
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def removeHooks(self, hooks):
        for hook in hooks:
            hook.remove()

#   ==================================================== Helpers ==================================================================


def setGradientsofModel(arch, net, text_input_tuple, device, hook_manager):
    if arch == "transformer":
        _setGradientsTransformerModel(
            net, text_input_tuple, device, hook_manager)
    elif arch == "cnn":
        _setGradientsCNNModel(net, text_input_tuple, device, hook_manager)
    else:
        raise ValueError(f"Model architecture {arch} not supported")

def _setGradientsCNNModel(net, input_for_model, device, hook_manager):
    # Insert hooks for capturing backward gradients of the CNN model
    net.zero_grad()
    all_layers = getAllLayers(net)
    all_hooks = [hook_manager.insertBackwardHook(
        layer) for layer in all_layers]

    net = net.to(device)
    img_input = input_for_model['pixel_values']
    # img_input = torch.tensor(img_input, dtype=torch.float32)

    outs = net(img_input.to(device))

    logits = outs  # Access the logits from the output object

    prob, predicted = torch.max(logits, dim=1)
    logits[0, predicted].backward()  # computing the gradients
    hook_manager.removeHooks(all_hooks)
    hook_manager.backward_hooks_storage.reverse()


def _setGradientsTransformerModel(net, text_input_tuple, device, hook_manager):
    # Insert hooks for capturing backward gradients of the transformer model
    net.zero_grad()
    all_layers = getAllLayers(net)
    all_hooks = [hook_manager.insertBackwardHook(
        layer) for layer in all_layers]

    net.to(device)

    # Assume text_input_tuple is already on the correct device and prepared
    text_input_tuple = {k: torch.tensor(v, device=device).unsqueeze(
        0) for k, v in text_input_tuple.items() if k in ["input_ids", "token_type_ids", "attention_mask"]}

    outs = net(**text_input_tuple)

    logits = outs.logits  # Access the logits from the output object

    prob, predicted = torch.max(logits, dim=1)
    predicted = predicted.cpu().detach().item()
    logits[0, predicted].backward()  # computing the gradients
    hook_manager.removeHooks(all_hooks)
    hook_manager.backward_hooks_storage.reverse()


def getAllLayers(net):
    layers = getAllLayersBert(net)
    # layers = layers[:len(layers)-2]

    # layers = [layers[-1]]

    return layers #[len(layers)-1:len(layers)]


def getAllLayersBert(net):
    layers = []
    for layer in net.children():
        # or isinstance(layer, (modeling_roberta.RobertaLayer, modeling_roberta.RobertaClassificationHead)) or isinstance(layer, modeling_bert.BertLayer) or isinstance(layer, modeling_bert.BertPooler):
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, transformers.pytorch_utils.Conv1D, modeling_bert.BertLayer)):
            layers.append(layer)
        elif len(list(layer.children())) > 0:
            temp_layers = getAllLayersBert(layer)
            layers = layers + temp_layers
    return layers


# def getAllLayersOpenAI(net):
#     layers = []
#     for layer in net.children():

#         # if isinstance(layer, torch.nn.Linear) or isinstance(layer, modeling_openai.Block) or isinstance(layer, modeling_bert.BertPooler):
#         if isinstance(layer, (torch.nn.Linear, transformers.pytorch_utils.Conv1D)) and len(list(layer.children())) == 0:
#             layers.append(layer)
#             # print(f"   -- layer : {layer}")
#         elif len(list(layer.children())) > 0:
#             print( f"> Type {type(layer)} || module : {layer}")
#             temp_layers = getAllLayersOpenAI(layer)
#             layers = layers + temp_layers
#         return layers


if __name__ == "__main__":
    name = 'google-bert/bert-base-cased'
    # name = 'google-bert/bert-large-cased'
    # name = 'openai-community/openai-gpt'

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=name, num_labels=14,)

    print(f"Model: {model}")

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    # Set pad_token to eos_token or add a new pad_token if eos_token is not available
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            tokenizer.add_special_tokens(
                special_tokens_dict={'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

    net = model
    all_layers = getAllLayers(net)

    print(all_layers)
    print(f'total layers: {len(all_layers)}')
    # Calculate the number of trainable parameters
    import torchvision
    model = torchvision.models.densenet121()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {trainable_params}')

