import os
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import read_audio, write_audio
from tqdm.contrib import tzip

def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(data_folder, "train.csv"),
        replacements={"data_root": data_folder},
    )
    datasets = [train_data]
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

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data

def add_noise(data_folder):
    # data_folder = "/home/hjyeee/Projects/AAI-project/LibriSpeech-SI"
    data_folder = data_folder
    output_folder = os.path.join(data_folder, "train-noisy")
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    overrides = {
        "data_folder": data_folder,
    }
    with open(hyperparams_file) as fin:
        hyperparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=output_folder,
        hyperparams_to_save=hyperparams_file,
        overrides=overrides,
    )

    train_data_folder = os.path.join(data_folder, "annotation")
    train_data = data_prep(train_data_folder, hyperparams)
    print(len(train_data))

    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset=train_data, batch_size=1
    )

    for batch in tzip(dataloader):
    # for batch in iter(dataloader):
        id = batch[0].id
        wav, wav_len = batch[0].sig
        # wav_noise, wav_len = batch[0].sig
        wav_noise = hyperparams["add_noise"](wav, wav_len)
        # save results on file
        for i, snt_id in enumerate(id):
            sp_folder = os.path.join(output_folder, snt_id.split("_")[0])
            if not os.path.exists(sp_folder):
                os.makedirs(sp_folder)
            filepath = (
                sp_folder + "/" + snt_id + "_noisy.flac"
            )
            write_audio(filepath, wav_noise[i], 16000)


if __name__ == "__main__":
    add_noise()
