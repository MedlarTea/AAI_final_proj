# LibriSpeech ASR with Transformers or Whisper models.
This folder contains the scripts to train a Transformer-based speech recognizer or the scripts to fine-tune the Whisper encoder-decoder model.

You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train_with_whisper.py hparams/train_hf_whisper.yaml

python train.py hparams/transformer.yaml

**If using a HuggingFace pre-trained model, please make sure you have "transformers"
installed in your environment (see extra-requirements.txt)**
# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 24-03-22 | transformer.yaml | 2.26 | [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | [GoogleDrive](https://drive.google.com/drive/folders/1sM3_PksmGQZMxXPibp7W7mQfPXFdHqc5?usp=sharing) | 1xA100 40GB |
| 06-12-23 | train_hf_whisper.yaml | 3.60 | Not Avail. | Not Avail. | 1xA100 40GB |

# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech
- https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech
- https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://drive.google.com/drive/folders/15uUZ21HYnw4KyOPW3tx8bLrS9RoBZfS7?usp=sharing

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
