#!/usr/bin/env/python3
"""This minimal example trains a speaker identification system based on
x-vectors. The encoder is based on TDNNs. The classifier is a MLP.
"""

import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import os.path as osp
import torch
import argparse
from xvector_brain import XvectorBrain

def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=osp.join(data_folder, "train_tr.csv"),
        replacements={"data_root": data_folder},
    )
    valid_clean_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=osp.join(data_folder, "train_dev_clean.csv"),
        replacements={"data_root": data_folder},
    )
    valid_noisy_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=osp.join(data_folder, "train_dev_noisy.csv"),
        replacements={"data_root": data_folder},
    )
    test_clean_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=osp.join(data_folder, "test.csv"),
        replacements={"data_root": data_folder},
    )
    test_noisy_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=osp.join(data_folder, "test-noisy.csv"),
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_clean_data, valid_noisy_data, test_clean_data, test_noisy_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.update_from_didataset(train_data, output_key="spk_id")
    label_encoder.update_from_didataset(valid_clean_data, output_key="spk_id")
    label_encoder.update_from_didataset(valid_noisy_data, output_key="spk_id")
    label_encoder.update_from_didataset(test_clean_data, output_key="spk_id")
    label_encoder.update_from_didataset(test_noisy_data, output_key="spk_id")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_clean_data, valid_noisy_data, test_clean_data, test_noisy_data


def main(device="cpu"):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default="/home/hjyeee/Projects/AAI-project/LibriSpeech-SI", type=str, help="data folder")
    # parser.add_argument('-c', '--ckpt', default="/home/hjyeee/Projects/AAI-project/speaker_identification/Example/results/1234/save/CKPT+2022-12-20+23-07-33+00/model.ckpt", type=str, help="data folder")

    args = parser.parse_args()

    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = osp.join(args.data_folder, "annotation")
    # ckpt_path = args.ckpt
    # data_folder = (experiment_dir / data_folder).resolve()

    overrides = {
        "data_folder": data_folder,
    }

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset creation
    _, valid_clean_data, valid_noisy_data, test_clean_data, test_noisy_data  = data_prep(data_folder, hparams)

    # Load the model
    xvect_brain = XvectorBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"]
    )
    print("------- Evaluating Valid Clean Data -------")
    xvect_brain.evaluate(valid_clean_data, max_key="all_acc",test_loader_kwargs=hparams["dataloader_options"])
    print("------- Evaluating Valid Noisy Data -------")
    xvect_brain.evaluate(valid_noisy_data, max_key="all_acc",test_loader_kwargs=hparams["dataloader_options"])

    
    # evaluate
    test_txt = osp.join(data_folder, "test_predictions.txt")
    file_p = open(test_txt, "w")
    predictions = {}
    print("------- Evaluating Test Clean Data -------")
    clean_predictions = xvect_brain.infer(test_clean_data, max_key="all_acc",test_loader_kwargs=hparams["dataloader_options"])
    predictions.update(clean_predictions)
    print("------- Evaluating Test Noisy Data -------")
    noisy_predictions = xvect_brain.infer(test_noisy_data, max_key="all_acc",test_loader_kwargs=hparams["dataloader_options"])
    predictions.update(noisy_predictions)
    print("------- Writing -------")
    for file_id in sorted(predictions.keys()):
        file_p.write("{} {}\n".format(file_id+".flac", "spk"+"{:03d}".format(int(predictions[file_id]))))
    file_p.close()
    print("------- Done -------")


    # model.eval()


if __name__ == "__main__":
    main(device="cuda")