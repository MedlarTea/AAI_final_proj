import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import os.path as osp
import torch
import argparse
from speechbrain.utils.distributed import run_on_main
from tqdm.contrib import tqdm

# Trains xvector model
class XvectorBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the speaker probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x_vect = self.modules.xvector_model(feats)
        outputs = self.modules.classifier(x_vect)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CE loss."
        predictions, lens = predictions
        spkid, spkid_lens = batch.spk_id_encoded
        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid, lens)
            self.acc_metric.append(batch.id, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            # Define function taking (prediction, target, length) for eval
            def accuracy_value(predict, target, lengths):
                """Computes Accuracy"""
                nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                    predict, target, lengths
                )
                acc = torch.tensor([nbr_correct / nbr_total])
                return acc

            self.acc_metric = sb.utils.metric_stats.MetricStats(
                metric=accuracy_value, n_jobs=1
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = stage_stats
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(
                stage, "error: %.2f" % self.error_metrics.summarize("average")
            )
            print(stage, "acc: %.2f" % self.acc_metric.summarize("average"))
            
    
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_clean_set=None,
        valid_noisy_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_clean_set is not None and not (
            isinstance(valid_clean_set, DataLoader)
            or isinstance(valid_clean_set, LoopedLoader)
        ):
            valid_clean_set = self.make_dataloader(
                valid_clean_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )
        if valid_noisy_set is not None and not (
            isinstance(valid_noisy_set, DataLoader)
            or isinstance(valid_noisy_set, LoopedLoader)
        ):
            valid_noisy_set = self.make_dataloader(
                valid_noisy_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            stage_stats = {"all_acc": 0, "clean_acc": 0, "noisy_acc": 0}
            print("----- Evaluate in clean valid set -----")
            self._fit_valid(valid_set=valid_clean_set, epoch=epoch, enable=enable)
            stage_stats["clean_acc"] = self.acc_metric.summarize("average")
            print("----- Evaluate in noisy valid set -----")
            self._fit_valid(valid_set=valid_noisy_set, epoch=epoch, enable=enable)
            stage_stats["noisy_acc"] = self.acc_metric.summarize("average")
            stage_stats["all_acc"] = (stage_stats["clean_acc"] + stage_stats["noisy_acc"]) / 2.0

            self.checkpointer.save_and_keep_only(
                    meta={"train_loss": self.train_loss,
                          "all_acc": stage_stats["all_acc"],
                          "clean_acc": stage_stats["clean_acc"],
                          "noisy_acc": stage_stats["noisy_acc"]},
                    num_to_keep=1,
                    max_keys=["all_acc"]
                )
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break
    def infer(self, test_set, max_key=None, min_key=None, progressbar=None, test_loader_kwargs={}):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )

        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.modules.eval()
        predictions = {}
        with torch.no_grad():
            # for batch in test_set:
            #     pass
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                outputs, lens = self.compute_forward(batch, stage=sb.Stage.TEST)
                spk_id_predicts = torch.argmax(outputs.squeeze(), dim=1)
                for i in range(len(batch.id)):
                    predictions.update({batch.id[i]: spk_id_predicts[i].cpu().numpy()})
        return predictions

