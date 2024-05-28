#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchmetrics

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# 7f5442d1-e5e5-11e9-9ce9-00505601122b 98274acb-959f-4faa-b638-69ed5d646df0

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=19, type=int, help="Number of epochs.")

parser.add_argument("--cle_dim", default=384, type=int, help="CLE embedding dimension.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--layers", default=2, type=int, help="Number of layers in rnn encoder.")
parser.add_argument("--rnn_dim", default=768, type=int, help="RNN layer dimension.")
parser.add_argument("--dictionary_dim", default=768, type=int, help="Dictionary tags embedding dimension.")
parser.add_argument("--dictionary_dropout", default=0.5, type=float, help="Dropout rate on the dictionary tags embedding.")
parser.add_argument("--we_dim", default=768, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.25, type=float, help="Mask words with the given probability.")
parser.add_argument("--clip", default=3, type=float, help="The max norm of gradients used for clipping.")
parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing to use.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate on the main rnn network.")

parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class TrainableModule(torch.nn.Module):
    """A simple Keras-like module for training with raw PyTorch.

    The module provides fit/evaluate/predict methods, computes loss and metrics,
    and generates both TensorBoard and console logs. By default, it uses GPU
    if available, and CPU otherwise. Additionally, it offers a Keras-like
    initialization of the weights.

    The current implementation supports models with either single input or
    a tuple of inputs; however, only one output is currently supported.
    """
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    from time import time as _time
    from tqdm import tqdm as _tqdm

    def configure(self, *, optimizer=None, schedule=None, loss=None, metrics={}, logdir=None, device="auto", clip: float | None = None):
        """Configure the module process.

        - `optimizer` is the optimizer to use for training;
        - `schedule` is an optional learning rate scheduler used after every batch;
        - `loss` is the loss function to minimize;
        - `metrics` is a dictionary of additional metrics to compute;
        - `logdir` is an optional directory where TensorBoard logs should be written;
        - `device` is the device to use; when "auto", `cuda` is used when available, `cpu` otherwise.
        """
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss, self.loss_metric = loss, torchmetrics.MeanMetric()
        self.metrics = torchmetrics.MetricCollection(metrics)
        self.logdir, self._writers = logdir, {}
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.to(self.device)
        self.clip = clip

    def load_weights(self, path, device="auto"):
        """Load the model weights from the given path."""
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save_weights(self, path):
        """Save the model weights to the given path."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        """Train the model on the given dataset.

        - `dataloader` is the training dataset, each element a pair of inputs and an output;
          the inputs can be either a single tensor or a tuple of tensors;
        - `dev` is an optional development dataset;
        - `epochs` is the number of epochs to train;
        - `callbacks` is a list of callbacks to call after each epoch with
          arguments `self`, `epoch`, and `logs`;
        - `verbose` controls the verbosity: 0 for silent, 1 for persistent progress bar,
          2 for a progress bar only when writing to a console.
        """
        for epoch in range(epochs):
            self.train()
            self.loss_metric.reset()
            self.metrics.reset()
            start = self._time()
            epoch_message = f"Epoch={epoch+1}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if verbose == 2 else not verbose)
            for xs, y in data_and_progress:
                xs, y = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(self.device)
                logs = self.train_step(xs, y)
                message = [epoch_message] + [f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()]
                data_and_progress.set_description(" ".join(message), refresh=False)
            if dev is not None:
                logs |= {"dev_" + k: v for k, v in self.evaluate(dev, verbose=0).items()}
            for callback in callbacks:
                callback(self, epoch, logs)
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("dev_")}, epoch + 1)
            self.add_logs("dev", {k[4:]: v for k, v in logs.items() if k.startswith("dev_")}, epoch + 1)
            verbose and print(epoch_message, "{:.1f}s".format(self._time() - start),
                              *[f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def train_step(self, xs, y):
        """An overridable method performing a single training step.

        A dictionary with the loss and metrics should be returned."""
        self.zero_grad()
        y_pred = self.forward(*xs)
        loss = self.loss(y_pred, y)
        loss.backward()
        with torch.no_grad():
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip)
            self.optimizer.step()
            self.schedule is not None and self.schedule.step()
            self.loss_metric.update(loss)
            self.metrics.update(y_pred, y)
            return {"loss": self.loss_metric.compute()} \
                | ({"lr": self.schedule.get_last_lr()[0]} if self.schedule else {}) \
                | self.metrics.compute()

    def evaluate(self, dataloader, verbose=1):
        """An evaluation of the model on the given dataset.

        - `dataloader` is the dataset to evaluate on, each element a pair of inputs
          and an output, the inputs either a single tensor or a tuple of tensors;
        - `verbose` controls the verbosity: 0 for silent, 1 for a single message."""
        self.eval()
        self.loss_metric.reset()
        self.metrics.reset()
        for xs, y in dataloader:
            xs, y = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(self.device)
            logs = self.test_step(xs, y)
        verbose and print("Evaluation", *[f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def test_step(self, xs, y):
        """An overridable method performing a single evaluation step.

        A dictionary with the loss and metrics should be returned."""
        with torch.no_grad():
            y_pred = self.forward(*xs)
            self.loss_metric.update(self.loss(y_pred, y))
            self.metrics.update(y_pred, y)
            return {"loss": self.loss_metric.compute()} | self.metrics.compute()

    def predict(self, dataloader, as_numpy=True):
        """Compute predictions for the given dataset.

        - `dataloader` is the dataset to predict on, each element either
          directly the input or a tuple whose first element is the input;
          the input can be either a single tensor or a tuple of tensors;
        - `as_numpy` is a flag controlling whether the output should be
          converted to a numpy array or kept as a PyTorch tensor.

        The method returns a Python list whose elements are predictions
        of the individual examples. Note that if the input was padded, so
        will be the predictions, which will then need to be trimmed."""
        self.eval()
        predictions = []
        for batch in dataloader:
            xs = batch[0] if isinstance(batch, tuple) else batch
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            batch = self.predict_step(xs)
            predictions.extend(batch.numpy(force=True) if as_numpy else batch)
        return predictions

    def predict_step(self, xs):
        """An overridable method performing a single prediction step."""
        with torch.no_grad():
            return self.forward(*xs)

    def writer(self, writer):
        """Possibly create and return a TensorBoard writer for the given name."""
        if writer not in self._writers:
            self._writers[writer] = self._SummaryWriter(os.path.join(self.logdir, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        """Log the given dictionary to TensorBoard with a given name and step number."""
        if logs and self.logdir:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    @staticmethod
    def keras_init(module):
        """Initialize weights using the Keras defaults."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                               torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.uniform_(module.weight, -0.05, 0.05)
        if isinstance(module, (torch.nn.RNNBase, torch.nn.RNNCellBase)):
            for name, parameter in module.named_parameters():
                "weight_ih" in name and torch.nn.init.xavier_uniform_(parameter)
                "weight_hh" in name and torch.nn.init.orthogonal_(parameter)
                "bias" in name and torch.nn.init.zeros_(parameter)
                if "bias" in name and isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
                    parameter.data[module.hidden_size:module.hidden_size * 2] = 1

class Model(TrainableModule):
    class MaskElements(torch.nn.Module):
        """A layer randomly masking elements with a given value."""
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs):
            if self.training and self._mask_probability:
                mask = torch.rand(size=inputs.shape, dtype=torch.float32, device=inputs.device)
                inputs[mask < self._mask_probability] = self._mask_value
            return inputs

    class EmbedCharacters(torch.nn.Module):
        """A layer converting character indices to embeddings."""
        def __init__(self, num_embeddings: int, embedding_dim: int, dropout: int = 0.5, padding_idx=MorphoDataset.PAD):
            super().__init__()
            self.padding_idx = padding_idx
            self._embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
            self._dropout = torch.nn.Dropout(dropout)
            self._char_rnn = torch.nn.GRU(embedding_dim, embedding_dim, bidirectional=True, batch_first=True)

        def forward(self, unique_forms, form_indices):
            # unique_forms: [num_unique_forms, max_form_length]
            # form_indices: [num_sentences, max_sentence_length]
            # [num_unique_forms, max_form_length, embedding_dim]
            cle = self._embedding(unique_forms)
            cle = self._dropout(cle)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                cle,
                unique_forms.ne(self.padding_idx).sum(dim=-1).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            # [2, num_unique_forms, embedding_dim]
            _, h_n = self._char_rnn(packed)
            # Concatenate the states of the forward and backward directions.
            # [num_unique_forms, 2 * embedding_dim]
            cle_states = torch.concat((h_n[0], h_n[1]), dim=-1)
            # [batch_size, max_sentence_length, 2 * embedding_dim]
            return torch.nn.functional.embedding(form_indices, cle_states)

    class EmbedDictionaryTags(torch.nn.Module):
        """A layer converting dictionary tag indices to embeddings."""
        def __init__(self, num_embeddings: int, embedding_dim: int, dropout: int = 0.5, padding_idx=MorphoDataset.PAD):
            super().__init__()

            self.padding_idx = padding_idx
            self._embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
            self._dropout = torch.nn.Dropout(dropout)
            self.tag_rnn = torch.nn.GRU(embedding_dim, embedding_dim, bidirectional=True, batch_first=True)

        def forward(self, dictionary_tag_ids: torch.Tensor):
            # dictionary_tag_ids: [batch_size, sequence_length, padded_num_tags]
            # [batch_size, sequence_length, padded_num_tags, embedding_dim]
            embeddings = self._embedding(dictionary_tag_ids)
            
            embeddings = self._dropout(embeddings)

            indices = dictionary_tag_ids.ne(self.padding_idx).sum(dim=-1).cpu()
            states = []
            for index, embedding in zip(indices, embeddings):
                if (index == 0).all():
                    state = torch.zeros((embedding.shape[0], embedding.shape[2]), dtype=embedding.dtype, device=embedding.device)
                else:
                    packed = torch.nn.utils.rnn.pack_padded_sequence(
                        embedding[index > 0],
                        index[index > 0],
                        batch_first=True,
                        enforce_sorted=False,
                    )
                    # [2, sequence_length, embedding_dim]
                    _, h_n = self.tag_rnn(packed)
                    # [sequence_length, embedding_dim]
                    state = torch.sum(h_n, dim=0)
                    state = torch.nn.functional.pad(state, (0, 0, 0, index.shape[0] - state.shape[0]))
                states.append(state)
            # [batch_size, sequence_length, embedding_dim]
            return torch.stack(states, dim=0)

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK)
        self._char_embedding = self.EmbedCharacters(len(train.forms.char_vocab), args.cle_dim)
        self._word_embedding = torch.nn.Embedding(len(train.forms.word_vocab), args.we_dim)
        self._dictionary_embedding = self.EmbedDictionaryTags(len(train.tags.word_vocab), args.dictionary_dim, args.dictionary_dropout)

        self._word_rnn = torch.nn.LSTM(
                args.we_dim + 2 * args.cle_dim + args.dictionary_dim,
                args.rnn_dim,
                num_layers=args.layers,
                bidirectional=True,
                batch_first=True,
            )
        
        self._conv_f = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=args.rnn_dim, out_channels=args.rnn_dim, kernel_size=12, stride=1, padding = "same"),
                )

        self._conv_b = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=args.rnn_dim, out_channels=args.rnn_dim, kernel_size=12, stride=1, padding = "same"),
                )

        self._output_layer = torch.nn.Conv1d(in_channels=args.rnn_dim, out_channels=len(train.tags.word_vocab), kernel_size=1, stride=1, padding = "same")

        self.apply(self.keras_init)

    def forward(self, form_ids: torch.Tensor, unique_forms: torch.Tensor, form_indices: torch.Tensor, dictionary_tag_ids: torch.Tensor) -> torch.Tensor:
        form_ids = self._word_masking(form_ids)

        # [batch_size, sequence_length, embedding_dim]
        hidden = self._word_embedding(form_ids)
        cle = self._char_embedding(unique_forms, form_indices)
        dictionary = self._dictionary_embedding(dictionary_tag_ids)

        hidden = torch.concat((hidden, cle, dictionary), dim=-1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden,
            form_ids.ne(MorphoDataset.PAD).sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        hidden, _ = self._word_rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, padding_value=MorphoDataset.PAD)

        hidden = hidden.view(hidden.size(0), hidden.size(1), 2, -1).permute(0, 3, 2, 1) # sum the two directions

        hidden_f = self._conv_f(hidden[:, :, 0, :])
        hidden_b = self._conv_b(hidden[:, :, 1, :])

        hidden = hidden_f + hidden_b

        hidden = self._bn(hidden)

        hidden = self._output_layer(hidden)

        return hidden


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)
    # tags = HierarchicalTags(morpho.train)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    model = Model(args, morpho.train)

    def prepare_tagging_data(example):
        form_ids = torch.tensor(morpho.train.forms.word_vocab.indices(example["forms"]), dtype=torch.long)
        tag_ids = torch.tensor(morpho.train.tags.word_vocab.indices(example["tags"]), dtype=torch.long)

        dictionary_analyses = [analyses.get(form) for form in example["forms"]]
        dictionary_tags = [[lemmatag.tag for lemmatag in analysis] for analysis in dictionary_analyses]
        # [sequence_length, num_tags_per_form]
        dictionary_tag_ids = [torch.tensor(morpho.train.tags.word_vocab.indices(tags), dtype=torch.long) for tags in dictionary_tags]

        return form_ids, example["forms"], dictionary_tag_ids, tag_ids

    train = morpho.train.transform(prepare_tagging_data)
    dev = morpho.dev.transform(prepare_tagging_data)
    test = morpho.test.transform(prepare_tagging_data)

    def prepare_batch(data):
        form_ids, forms, dictionary_tag_ids, tag_ids = zip(*data)
        form_ids = torch.nn.utils.rnn.pad_sequence(form_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        unique_strings, unique_forms, forms_indices = morpho.train.cle_batch(forms)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)

        max_sentence = max(len(sentence) for sentence in dictionary_tag_ids)
        max_tags = max(len(tags) for sentence in dictionary_tag_ids for tags in sentence)

        dictionary_tag_ids = list(dictionary_tag_ids)
        for i_s in range(len(dictionary_tag_ids)):
            sentence = dictionary_tag_ids[i_s]
            for i_t in range(len(sentence)):
                tags = sentence[i_t]
                sentence[i_t] = torch.nn.functional.pad(tags, (0, max_tags - len(tags)), value=MorphoDataset.PAD)
            sentence = torch.stack(sentence, dim=0)
            dictionary_tag_ids[i_s] = torch.nn.functional.pad(sentence, (0, 0, 0, max_sentence - len(sentence)), value=MorphoDataset.PAD)
        # [batch_size, padded_sequence_length, padded_num_tags]
        dictionary_tag_ids = torch.stack(dictionary_tag_ids, dim=0).to(torch.long)

        return (form_ids, unique_forms, forms_indices, dictionary_tag_ids), tag_ids

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, collate_fn=prepare_batch)
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size, collate_fn=prepare_batch)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train))
    model.configure(
        optimizer=optimizer,
        schedule=scheduler,
        # The loss expects the input to be of shape `[batch_size, num_tags, sequence_length]`.
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD, label_smoothing=args.smoothing),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=len(morpho.train.tags.word_vocab), ignore_index=morpho.PAD)},
        logdir=args.logdir,
        clip=args.clip,
    )

    logs = model.fit(train, epochs=args.epochs, dev=dev)

    torch.save(model.state_dict(), "tagger_competition.pt")

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set; update the following code
        # if you use other output structure than in tagger_we.
        predictions = model.predict(test)

        for predicted_tags, forms in zip(predictions, morpho.test.forms.strings):
            for predicted_tag in np.argmax(predicted_tags[:, :len(forms)], axis=0):
                print(morpho.train.tags.word_vocab.string(predicted_tag), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
