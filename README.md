# Speaker Identification

## File directory
```
AAI_project
|
---speaker_identification (This directory)
---LibriSpeech-SI
---speechbrain

```


## Prerequisite

### Virtual environment
```bash
# It's better to use a conda environment, or anyelse
conda create-n si python=3.8
```

### Our codes
```bash
git clone https://github.com/MedlarTea/AAI_final_proj
```

### Datatset
Download from [bb](https://bb.sustech.edu.cn/bbcswebdav/courses/CSE5001-30008993-2022FA/LibriSpeech-SI.tar.gz), unzip as `LibriSpeech-SI`

### Install Speechbrain
```bash
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
cp ../AAI_final_proj/requirements.txt .
pip install -r requirements.txt
pip install --editable .
```

## How to run
```bash
# Create datasets
cd Example
# python create_dataset.py --data_folder /home/hjyeee/Projects/AAI-project/LibriSpeech-SI
python create_dataset.py --data_folder {dataset path}

# Run training and testing
# change "hyperparams.yaml" to finetune the model
python example_xvector_experiment.py --data_folder {dataset path}

# Infer the test data and write it to "annotation/test_predictions.txt"
python test.py --data_folder {dataset path}
```



