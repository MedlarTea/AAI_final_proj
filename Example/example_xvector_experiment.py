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
from speechbrain.utils.distributed import run_on_main
from xvector_brain import XvectorBrain

### (lzj: add for experiments)
import time
import nni


### (lzj: for auto tuning)
g_params = {
    'lr': 0.001,
    'batch_size': 32,
    # 'xvector_tdnn_ch0': 512,
    # 'xvector_tdnn_ch1': 512,
    # 'xvector_tdnn_ch2': 512,
    # 'xvector_tdnn_ch3': 512,
    # 'xvector_tdnn_ch4': 1500,     
}
optimized_params = nni.get_next_parameter()
g_params.update(optimized_params)
# print(g_params)




## (lzj: test) Try to modified param here
def adjust_param(hparams, param_struct):

    ### file paths
    experiment_time = time.strftime("Date_%Y_%m_%d_Time_%H_%M_%S", time.gmtime())
    output_str = experiment_time
    experiment_dir = pathlib.Path(__file__).resolve().parent
    output_folder = experiment_dir / "results" / output_str
    wer_file = output_folder / "wer.txt"
    save_folder = output_folder / "save"
    train_log = output_folder / "train_log.txt"

    output_folder.mkdir()
    save_folder.mkdir()
    

    hparams["output_folder"] = output_folder
    hparams["wer_file"] = wer_file
    ## save_folder & checkpointer.checkpoints_dir
    hparams["save_folder"] = save_folder
    hparams["checkpointer"].checkpoints_dir = save_folder
    ## train_log & train_logger.save_file
    hparams["train_log"] = train_log
    hparams["train_logger"].save_file = train_log
    print(f'output_folder: {output_folder}')

    ### modify params
    # N_epochs & epoch_counter.limit
    # hparams["N_epochs"] = 200 
    # hparams["epoch_counter"].limit = 200 
    # print(f'hparams["N_epochs"]: {hparams["N_epochs"]}')
    # print(f'hparams["epoch_counter"].limit: {hparams["epoch_counter"].limit}')     
     
    # lr & opt_class.lr
    hparams["lr"] = param_struct["lr"]
    hparams["opt_class"].lr = param_struct["lr"]
    print(f'hparams["lr"]: {hparams["lr"]}')
    print(f'hparams["opt_class"].lr: {hparams["opt_class"].lr}')
    # dataloader_options.batch_size 
    # hparams["dataloader_options"].batch_size = param_struct["batch_size"]
    hparams["dataloader_options"]["batch_size"] = param_struct["batch_size"]   
    print(f'hparams["dataloader_options"]["batch_size"]: {hparams["dataloader_options"]["batch_size"]}')   
    # xvector_model.tdnn_channels 



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

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_clean_data, valid_noisy_data, test_clean_data, test_noisy_data


def main(device="cpu"):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default="/data/zijun/Workspaces/CourseProj_ws/AAI/LibriSpeech-SI", type=str,
    help="data folder")
    args = parser.parse_args()

    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = osp.join(args.data_folder, "annotation")
    # data_folder = (experiment_dir / data_folder).resolve()

    overrides = {
        "data_folder": data_folder,
    }

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    ### (lzj: for auto tuning)
    adjust_param(hparams, g_params)

 
    # Dataset creation
    train_data, valid_clean_data, valid_noisy_data, test_clean_data, test_noisy_data = data_prep(data_folder, hparams)

    # Trainer initialization
    xvect_brain = XvectorBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"]
    )

    # Training/validation loop
    xvect_brain.fit(
        hparams["epoch_counter"],
        train_data,
        valid_clean_data,
        valid_noisy_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # get the test set prediction


    # Evaluation is run separately (now just evaluating on valid data)
    ### (lzj: add avg_eval_loss)
    avg_eval_loss = xvect_brain.evaluate(valid_clean_data, max_key="all_acc", test_loader_kwargs=hparams["dataloader_options"])
    avg_eval_acc = xvect_brain.acc_metric.summarize("average")

   
    print(f'Average eval loss: {avg_eval_loss}')    
    print(f'Average eval acc: {avg_eval_acc}')
    nni.report_final_result(avg_eval_acc) 
    
    # Check if model overfits for integration test
    # (lzj: comment this due to a bug)
    # assert xvect_brain.train_loss < 0.2


if __name__ == "__main__":
    main(device="cuda:1")


def test_error(device):
    main(device)
