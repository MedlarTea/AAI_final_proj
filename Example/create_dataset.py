import os
import csv
import random
from collections import Counter
import logging
import torchaudio
from tqdm.contrib import tzip
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
)
from add_noise_lib import add_noise
import argparse

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_prepare.pkl"
SAMPLERATE = 16000
SEED = 123

def create_csv(
    save_folder, wav_lst, split, select_n_sentences,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id"]]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for index, wav_file in enumerate(tzip(wav_lst)):
        wav_file = wav_file[0]

        # in our case, its name, e.g., spk001_002.flac
        snt_id = wav_file.split("/")[-1].replace(".flac", "")
        spk_id = snt_id.split("_")[0]
        

        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE

        csv_line = [
            snt_id,
            str(duration),
            wav_file,
            spk_id
        ]

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1

        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)

def create_noise_csv(save_folder, wav_lst):
    """
    Create the dataset csv file given a list of wav files.
    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, "noise.csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "wav_format", "wav_opts"]]

    # Processing all the wav files in wav_lst
    snt_cnt = 0
    for wav_file in tzip(wav_lst):
        wav_file = wav_file[0]
        id = "noise_{:03d}".format(snt_cnt)
        if wav_file.endswith(".wav"):
            wav_format = "wav"
        elif wav_file.endswith(".flat"):
            wav_format = "flat"
        else:
            raise("Undown format")
        

        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE

        csv_line = [
            id,
            str(duration),
            wav_file,
            wav_format,
            ""
        ]

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1


    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)

def prepare_librispeech(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    skip_prep=False,
):
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # create csv files for each split
    for split_index in range(len(splits)):

        split = splits[split_index]

        wav_lst = sorted(get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"]
        ))

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_csv(save_folder, wav_lst, split, n_sentences)
    
    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_libri + ".csv" for split_libri in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
        )

def split_tr_dev(data_folder, clean_csv_filename, noisy_csv_filename, split_ratio):
    # Reading lexicon.csv
    train_clean_csv_path = os.path.join(data_folder, clean_csv_filename)
    train_noisy_csv_path = os.path.join(data_folder, noisy_csv_filename)
    with open(train_clean_csv_path, "r") as f:
        all_clean_lines = f.readlines()
    with open(train_noisy_csv_path, "r") as f:
        all_noisy_lines = f.readlines()
    # Remove header
    all_clean_lines = all_clean_lines[1:]
    all_noisy_lines = all_noisy_lines[1:]
    print("Total clean num: {}".format(len(all_clean_lines)))
    print("Total noisy num: {}".format(len(all_noisy_lines)))

    # Shuffle entries
    random.seed(SEED)
    random.shuffle(all_clean_lines)
    random.shuffle(all_noisy_lines)

    # Selecting lines
    header = "ID,duration,wav,spk_id\n"

    tr_clean_snts = int(0.01 * split_ratio[0] * len(all_clean_lines))
    train_clean_lines = [header] + all_clean_lines[0:tr_clean_snts]
    print("Train clean num: {}".format(len(train_clean_lines)-1))
    valid_clean_lines = [header] + all_clean_lines[tr_clean_snts : ]
    print("Valid clean num: {}".format(len(valid_clean_lines)-1))

    tr_noisy_snts = int(0.01 * split_ratio[0] * len(all_noisy_lines))
    train_noisy_lines = all_noisy_lines[0:tr_noisy_snts]
    print("Train noisy num: {}".format(len(train_noisy_lines)))
    valid_noisy_lines = [header] + all_noisy_lines[tr_noisy_snts : ]
    print("Valid noisy num: {}".format(len(valid_noisy_lines)-1))


    # valid_snts = int(0.01 * split_ratio[1] * len(all_lines))
    # valid_lines = [header] + all_lines[tr_snts : ]
    # print("Valid num: {}".format(len(valid_lines)-1))
    # test_lines = [header] + all_lines[tr_snts + valid_snts :]

    # Saving files
    with open(os.path.join(data_folder, "train_tr.csv"), "w") as f:
        f.writelines(train_clean_lines + train_noisy_lines)

    with open(os.path.join(data_folder, "train_dev_clean.csv"), "w") as f:
        f.writelines(valid_clean_lines)
    with open(os.path.join(data_folder, "train_dev_noisy.csv"), "w") as f:
        f.writelines(valid_noisy_lines)

def prepare_noise(data_folder, save_folder):
    split = "noise"
    noise_lst = sorted(get_all_files(
            os.path.join(data_folder, split), match_and=[".wav"]
        ))
    create_noise_csv(save_folder, noise_lst)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default="/home/hjyeee/Projects/AAI-project/LibriSpeech-SI", type=str,
    help="data folder")
    args = parser.parse_args()

    data_folder = args.data_folder
    tr_splits = ['train']
    dev_splits = []
    te_splits = ['test', 'test-noisy']
    save_folder = os.path.join(data_folder, "annotation") # for saving csv files

    # create clean trainset and testsets
    print("---------- Creating Trainset and Testsets -----------")
    prepare_librispeech(data_folder, save_folder, tr_splits, dev_splits, te_splits)

    # create noise.csv  
    print("---------- Creating noise.csv -----------")
    prepare_noise(data_folder, save_folder)

    # create noise trainset
    print("---------- Creating Noisy Trainset -----------")
    add_noise(data_folder)
    prepare_librispeech(data_folder, save_folder, ['train-noisy'])

    # split trainset to "train_tr.csv" with clean and non-clean trainsets; "train_clean_dev" with clean validset; "train_noise_dev" with non-clean validset
    split_tr_dev(save_folder, "train.csv", "train-noisy.csv", [80,20])

    