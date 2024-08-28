"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import gc

import torch
import torchvision
from torch.utils.data import DataLoader
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import logging
from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer,  AutoTokenizer, Phi3ForSequenceClassification   # type: ignore
from transformers import DefaultDataCollator


class CNNTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        batch_inputs = inputs.get("pixel_values")
        # print(f"batch_inputs: {batch_inputs}")
        outputs = model(batch_inputs)
        logits = outputs
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss = None
        labels = None
        logits = None
        with torch.no_grad():
            logits = model(inputs['pixel_values'])
            if 'labels' in inputs:
                labels = inputs.get("labels")
                loss = torch.nn.CrossEntropyLoss()(logits, labels)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def _compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct_predictions = predictions == labels
    incorrect_predictions = ~correct_predictions

    correct_indices = np.where(correct_predictions)[0]
    correct_indices = torch.from_numpy(correct_indices)

    incorrect_indices =  np.where(incorrect_predictions)[0]
    incorrect_indices = torch.from_numpy(incorrect_indices)


    d = {
        "accuracy": metric.compute(predictions=predictions, references=labels),
        "correct_indices": correct_indices,
        "actual_labels": labels,
        'incorrect_indices':incorrect_indices,
        'predicted_labels': predictions
    }
    return d


def _get_inputs_labels_from_batch(batch):
    if "pixel_values" in batch:
        return batch["pixel_values"], batch["label"]
    else:
        x, y = batch
        return x, y


def initialize_model(name, cfg_dataset):
    """Initialize the model with the given name."""
    model_dict = {"model": None, "num_classes": cfg_dataset.num_classes}

    if name in ['squeezebert/squeezebert-uncased', 'openai-community/openai-gpt', "Intel/dynamic_tinybert", "google-bert/bert-base-cased", 'microsoft/MiniLM-L12-H384-uncased', 'distilbert/distilbert-base-uncased']:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=name, num_labels=cfg_dataset.num_classes,
        )

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        # Set pad_token to eos_token or add a new pad_token if eos_token is not available
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id

        model_dict['model'] = model.cpu()
    elif name == "meta-llama/Llama-2-7b-hf":
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=name, num_labels=cfg_dataset.num_classes,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        # Set pad_token to eos_token or add a new pad_token if eos_token is not available
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id

        model_dict['model'] = model.cpu()
    elif name == 'microsoft/Phi-3-mini-4k-instruct':
        # net = PhiForSequenceClassification.from_pretrained(
        #     pretrained_model_name_or_path=name, num_labels=cfg_dataset.num_classes,
        # )
        model = Phi3ForSequenceClassification.from_pretrained(
            name, device_map="auto", num_labels=cfg_dataset.num_classes)
        # net = model.cpu()  # type: ignore
        model_dict['model'] = model.cpu()
        # Set pad_token to eos_token or add a new pad_token if eos_token is not available
        # if tokenizer.pad_token is None:
        #     if tokenizer.eos_token is not None:
        #         tokenizer.pad_token = tokenizer.eos_token
        #         model.config.pad_token_id = tokenizer.pad_token_id
        #     else:
        #         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #         model.resize_token_embeddings(len(tokenizer))
        #         model.config.pad_token_id = tokenizer.pad_token_id

    elif name in ["facebook/wav2vec2-base", "openai/whisper-tiny"]:
        # Model training
        model = AutoModelForAudioClassification.from_pretrained(
            pretrained_model_name_or_path=name, num_labels=cfg_dataset.num_classes)
        model_dict['model'] = model.cpu()
        processor = AutoProcessor.from_pretrained(name)
        # model_dict['audio_feature_extractor'] = AutoFeatureExtractor.from_pretrained(name)
        model_dict['audio_feature_extractor'] = processor.feature_extractor

    elif name.find("resnet") != -1:
        model = None
        if "resnet18" == name:
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif "resnet34" == name:
            model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        elif "resnet50" == name:
            model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        elif "resnet101" == name:
            model = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        elif "resnet152" == name:
            model = torchvision.models.resnet152(weights="IMAGENET1K_V1")

        if cfg_dataset.channels == 1:
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()

    elif name == "densenet121":
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        if cfg_dataset.channels == 1:
            logging.info("Changing the first layer of densenet model the model to accept 1 channel")
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()
    elif name == "vgg16":
        model = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        if cfg_dataset.channels == 1:
            model.features[0] = torch.nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(
            num_ftrs, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()
    else:
        raise ValueError(f"Model {name} not supported")

    return model_dict


# ---------------------- CNN Training  ------------------------


def _train_cnn(tconfig):
    """Train the network on the training set."""
    trainloader = DataLoader(
        tconfig["train_data"], batch_size=tconfig["batch_size"])
    net = tconfig["model_dict"]["model"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=tconfig["lr"])
    net.train()
    net = net.to(tconfig["device"])
    epoch_loss = 0
    epoch_acc = 0
    for _epoch in range(tconfig["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = _get_inputs_labels_from_batch(batch)
            images, labels = images.to(
                tconfig["device"]), labels.to(tconfig["device"])

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            images = images.cpu()
            labels = labels.cpu()
            gc.collect()
        epoch_loss /= total
        epoch_acc = correct / total
    net = net.cpu()
    gc.collect()
    return {"train_loss": epoch_loss, "train_accuracy": epoch_acc}


def _test_cnn(net, test_data, device):
    """Evaluate the network on the entire test set."""
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=512, shuffle=False, num_workers=4
    )

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net = net.to(device)
    with torch.no_grad():
        for batch in testloader:
            images, labels = _get_inputs_labels_from_batch(batch)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            images = images.cpu()
            labels = labels.cpu()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    net = net.cpu()
    return {"eval_loss": loss, "eval_accuracy": accuracy}


def _train_cnn_hf_trainer(tconfig):

    net = tconfig["model_dict"]["model"]
    # testing_args = TrainingArguments(
    #     logging_strategy="steps",
    #     output_dir="storage_of_exps",
    #     do_train=True,
    #     do_eval=False,  # Enable evaluation
    #     per_device_eval_batch_size=tconfig["batch_size"],
    #     disable_tqdm=True,  # Enable tqdm progress bar
    #     remove_unused_columns=False,
    #     report_to="none",
    # )

    model_dict = tconfig["model_dict"]
    net = model_dict["model"]
    net = net.to(tconfig["device"])

    fp16 = True
    if tconfig['device'].type == 'cpu':
        fp16 = False

    training_args = TrainingArguments(
        learning_rate=tconfig["lr"],
        output_dir='training_output',
        lr_scheduler_type="constant",  # Set learning rate scheduler to constant
        num_train_epochs=tconfig["epochs"],
        eval_strategy="no",
        per_device_train_batch_size=tconfig["batch_size"],
        per_device_eval_batch_size=tconfig['batch_size'],
        fp16=fp16,
        disable_tqdm=True,
        report_to="none",
        remove_unused_columns=False,

    )

    trainer = CNNTrainer(model=net, args=training_args, train_dataset=tconfig["train_data"], compute_metrics=_compute_metrics,
                         tokenizer=None,  # Not needed for CNNs
                         data_collator=None  # Not needed for CNNs
                         )  # type: ignore

    # Evaluate the model on the test dataset
    trainer.train()
    r = trainer.evaluate(eval_dataset=tconfig["train_data"])
    net.eval()
    net = net.cpu()
    return {"train_loss": r['eval_loss'], "train_accuracy": r['eval_accuracy']['accuracy']}


def _test_cnn_hf_trainer(gm_dict, central_server_test_data, batch_size):
    net = gm_dict['model']
    logging.debug("Evaluating cnn model")
    testing_args = TrainingArguments(
        logging_strategy="steps",
        output_dir="storage_of_exps",
        do_train=False,  # Disable training
        do_eval=True,  # Enable evaluation
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True,  # Enable tqdm progress bar
        remove_unused_columns=False,
        report_to="none",
    )

    tester = CNNTrainer(
        model=net,
        args=testing_args,
        # Ensure it uses the correct metrics for evaluation
        compute_metrics=_compute_metrics,
        data_collator=DefaultDataCollator()
    )

    logging.debug(f'lenght of eval dataset: {len(central_server_test_data)}')
    # Evaluate the model on the test dataset
    r = tester.evaluate(eval_dataset=central_server_test_data)
    net = net.cpu()
    return r


# ---------------------- Transformer Training  ------------------------


def _train_transformer(tconfig):
    """Train the transformer model."""

    model_dict = tconfig["model_dict"]
    net = model_dict["model"]
    net = net.to(tconfig["device"])

    training_args = TrainingArguments(
        output_dir='training_output',
        lr_scheduler_type="constant",  # Set learning rate scheduler to constant
        num_train_epochs=tconfig["epochs"],
        eval_strategy="no",
        per_device_train_batch_size=tconfig["batch_size"],
        per_device_eval_batch_size=tconfig['batch_size'],
        fp16=True,
        disable_tqdm=True,
        report_to="none"
    )

    # training_args = TrainingArguments(output_dir="test_trainer", lr_scheduler_type="constant", warmup_ratio=0.1, max_grad_norm=0.3, per_device_train_batch_size=batch_size,
    #                                   per_device_eval_batch_size=batch_size, num_train_epochs=local_epochs, weight_decay=0.001, evaluation_strategy="no",   fp16=True, gradient_checkpointing=True, )

    trainer = Trainer(model=net, args=training_args, train_dataset=tconfig["train_data"],
                      eval_dataset=tconfig["train_data"], compute_metrics=_compute_metrics,)  # type: ignore

    if 'audio_feature_extractor' in model_dict:
        trainer.tokenizer = model_dict['audio_feature_extractor']

    trainer.train()

    r = trainer.evaluate(eval_dataset=tconfig["train_data"])

    net = net.cpu()
    # type: ignore
    return {"train_loss": r['eval_loss'], "train_accuracy": r['eval_accuracy']['accuracy']}


def _test_transformer_model(args):
    logging.debug("Evaluating transformer model")
    model_dict, central_server_test_data, batch_size = args[
        "model_dict"], args["test_data"], args["batch_size"]
    net = model_dict['model']
    testing_args = TrainingArguments(logging_strategy="steps", output_dir="storage_of_exps", do_train=False,
                                     do_eval=True,   per_device_eval_batch_size=batch_size, disable_tqdm=True,   report_to="none")
    tester = Trainer(model=net, args=testing_args,
                     compute_metrics=_compute_metrics, eval_dataset=central_server_test_data)

    if 'audio_feature_extractor' in model_dict:
        tester = Trainer(model=net, args=testing_args,
                         compute_metrics=_compute_metrics, eval_dataset=central_server_test_data, tokenizer=model_dict['audio_feature_extractor'])

    logging.debug(f'lenght of eval dataset: {len(central_server_test_data)}')

    r = tester.evaluate()
    # r["eval_accuracy"] = r["eval_accuracy"]["accuracy"]  # type: ignore
    net = net.cpu()
    return r


def global_model_eval(arch, global_net_dict, server_testdata, batch_size=32):
    """Evaluate the global model on the server test data."""
    d = {}
    if arch == "cnn":
        # d = _test_cnn_pl_trianer(
        #     global_net_dict,
        #     central_server_test_data=server_testdata,
        #     batch_size=batch_size,
        # )[0]
        d = _test_cnn(global_net_dict["model"],
                      test_data=server_testdata, device="cuda")
    elif arch == "transformer":
        d = _test_transformer_model(
            {'model_dict': global_net_dict, 'test_data': server_testdata, 'batch_size': batch_size})

    return {
        "loss": d["eval_loss"],
        "accuracy": d["eval_accuracy"],
    }


def test_neural_network(arch, global_net_dict, server_testdata, batch_size=32):
    """Evaluate the global model on the server test data."""
    d = {}
    if arch == "cnn":
        d = _test_cnn_hf_trainer(
            global_net_dict,
            central_server_test_data=server_testdata,
            batch_size=batch_size,
        )
    elif arch == "transformer":
        d = _test_transformer_model(
            {'model_dict': global_net_dict, 'test_data': server_testdata, 'batch_size': batch_size})
    else:
        raise ValueError(f"Architecture {arch} not supported")

    d['loss'] = d['eval_loss']
    d['accuracy'] = d['eval_accuracy']['accuracy']
    return d


def train_neural_network(tconfig):
    """Train the neural network."""

    train_dict = {}

    if tconfig["arch"] == "cnn":
        train_dict = _train_cnn(tconfig)
        # train_dict = _train_cnn_hf_trainer(tconfig)

    elif tconfig["arch"] == "transformer":
        train_dict = _train_transformer(tconfig)
        print(f"--------------> train_dict: {train_dict}")
    else:
        raise ValueError(f"Architecture {tconfig['arch']} not supported")

    return train_dict
