# Fast Word Error Rate Estimation (Fe-WER, Fast WER-Estimation)
Minimum implementation of Word Error Rate estimation models from [Park et al. (2025)](https://arxiv.org/abs/2310.08225)

# Task List
- [x] Download TEDLIUM3 using torchaudio
- [x] Feature extraction (HuBERT, XLM-R)
- [x] Transcribe speech using Automatic Speech Recognition (ASR) models (Whisper/HuBERT)
- [x] Generate labels for Character Error Rate (CER) and Word Error Rate (WER)
- [x] Train a WER estimation model
- [x] Evaluate the model

# Envirnonment
## Computing
- Distributed Linux clusters with SLURM
## Requirement
- kaldiio
- nemo_text_processing
- numpy
- scipy
- sklearn
- speechbrain
- torch
- torchaudio
- whisper
## Set a directory for the project
- export PROJECT_DIR=*path_to_this_repository*

# Download a dataset: TED-LIUM corpus release 3
- download_tedlium3.py
- check_tedlium3.py

# Generate datasets
- generate_data.py

| Dataset   | # of segments   | Duration     |
| --------- | --------------- | ------------ |
| tl3_dev   | 591 segments    | 1.71 hours   |
| tl3_test  | 1469 segments   | 3.05 hours   |
| tl3_train | 268263 segments | 453.81 hours |
- examples of outputs
```
# data.audmap
TL3-MarianoSigman_2016_1:1294-2267 1 12.94 22.67 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:229-278 1 22.9 27.8 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:2737-2997 1 27.37 29.97 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:2996-4002 1 29.96 40.02 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:3959-4431 1 39.59 44.31 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:4439-5329 1 44.39 53.29 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:5478-6445 1 54.78 64.45 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:6402-6659 1 64.02 66.59 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:6617-7111 1 66.17 71.11 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
TL3-MarianoSigman_2016_1:714-7997 1 71.4 79.97 ${PROJECT_DIR}/datasets/TEDLIUM_release-3/legacy/train/sph/MarianoSigman_2016.sph
```
```
# data.stm
TL3-MarianoSigman_2016 1 MarianoSigman_2016 12.94 22.67 <id=1:1294-2267,dset=train> we have historical records that allow us to know how the ancient greeks dressed how they lived how they fought but how did they think
TL3-MarianoSigman_2016 1 MarianoSigman_2016 22.9 27.8 <id=1:229-278,dset=train> one natural idea is that the deepest aspects of human thought
TL3-MarianoSigman_2016 1 MarianoSigman_2016 27.37 29.97 <id=1:2737-2997,dset=train> our ability to imagine to be
TL3-MarianoSigman_2016 1 MarianoSigman_2016 29.96 40.02 <id=1:2996-4002,dset=train> conscious to dream have always been the same another possibility is that the social transformations that have shaped our culture
TL3-MarianoSigman_2016 1 MarianoSigman_2016 39.59 44.31 <id=1:3959-4431,dset=train> may have also changed the structural columns of human thought
TL3-MarianoSigman_2016 1 MarianoSigman_2016 44.39 53.29 <id=1:4439-5329,dset=train> we may all have different opinions about this actually it is a long standing philosophical debate but is this question even amenable to science
TL3-MarianoSigman_2016 1 MarianoSigman_2016 54.78 64.45 <id=1:5478-6445,dset=train> here i would like to propose that in the same way we can reconstruct how the ancient greek cities looked just based on a few bricks
TL3-MarianoSigman_2016 1 MarianoSigman_2016 64.02 66.59 <id=1:6402-6659,dset=train> that the writings of a culture
TL3-MarianoSigman_2016 1 MarianoSigman_2016 66.17 71.11 <id=1:6617-7111,dset=train> are the archeological records the fossils of human thought
TL3-MarianoSigman_2016 1 MarianoSigman_2016 71.4 79.97 <id=1:714-7997,dset=train> and in fact doing some form of psychological analysis of some of the most ancient books of human culture julian jaynes came
```

# Extract features
- installation of Whisper's normaliser
```
/share/mini1/res/t/asr/multi/multi-en/acsw/selft/repositories/whisper/whisper/normalizers/english.py
#s = self.standardize_numbers(s) # comment out this line
```
- feature_extraction.py
- example
```
export TOTAL_JOBS=10
export JOB_NUMBER=1 # 1..10
export DATASET_NAME=tl3_train # or tl3_dev tl3_test
export FEATURE_NAME=hubert # or xlmr
python -u feature_extraction.py \
--total_jobs    ${TOTAL_JOBS} \
--job_number    ${JOB_NUMBER} \
--dataset_name  ${DATASET_NAME} \
--feature_name  ${FEATURE_NAME} \
--training_type pt \
--sample_rate   16000 \
--feature_layer 24 \
--feature_level sequence \
--feature_size  large
```
```
# Merge .scp files
cat hubert-pt-24-sequence.large_*.scp > hubert-pt-24-sequence.large.scp
cat xlmr-pt-24-sequence.large_*.scp > xlmr-pt-24-sequence.large.scp
```

# Transcribe datasets
- transcription.py
- example
```
export TOTAL_JOBS=10
export JOB_NUMBER=1 # 1..10
export DATASET_NAME=tl3_train # or tl3_dev tl3_test
export MODEL_NAME=whisper
export MODEL_SIZE=large
python -u transcription.py \
--total_jobs    ${TOTAL_JOBS} \
--job_number    ${JOB_NUMBER} \
--dataset_name  ${DATASET_NAME} \
--sample_rate   16000 \
--model_name    ${MODEL_NAME} \
--model_size    ${MODEL_SIZE}
```

# Generate labels
- generate_labels.py
- example
```
exp[ort DATASET_NAME=tl3_train # or tl3_dev tl3_test
exp[ort MODEL_NAME=whisper
exp[ort MODEL_SIZE=large
python -u generate_labels.py \
--dataset_name  ${DATASET_NAME} \
--model_name    ${MODEL_NAME} \
--model_size    ${MODEL_SIZE}
```

# Train a WER estimation model
- training.py
- example
```
EXPERIMENT_NAME=Fe-WER
LAYER1_SIZE=600
LAYER2_SIZE=32
LEARNING_RATE=0.0007
python -u training.py \
--base_path ${PROJECT_DIR} \
--train_dataset_name tl3_train \
--valid_dataset_name tl3_dev \
--hypothesis_name whisper_large \
--utterance_encoder_name hubert-pt-24-sequence.large \
--transcript_encoder_name xlmr-pt-24-sequence.large \
--batch_size 64 \
--num_workers 0 \
--max_duration 10 \
--model_path ${PROJECT_DIR}/models/${EXPERIMENT_NAME}_${LAYER1_SIZE}_${LAYER2_SIZE}_${LEARNING_RATE} \
--layer_sizes 2048 ${LAYER1_SIZE} ${LAYER2_SIZE} 1 \
--dropout 0.1 \
--learning_rate ${LEARNING_RATE} \
--max_iteration 15 \
--max_epochs 200 \
--early_stop 40
```

# Evaluate a WER estimation model
- evaluation.py
- example
```
python -u evaluation.py \
--base_path ${PROJECT_DIR}
--test_dataset_name tl3_test
--hypothesis_name whisper_large
--utterance_encoder_name hubert-pt-24-sequence.large
--transcript_encoder_name xlmr-pt-24-sequence.large
--batch_size 64
--num_workers 0
--max_duration 10
--model_path ${PROJECT_DIR}/models/Fe-WER_600_32_0.0007
--layer_sizes 2048 600 32 1
--exp_path ${PROJECT_DIR}/scripts/evaluation_Fe-WER_600_32_0.0007
```

# Result                                
- hyperparameters
```
layer_size: 600 32
learning rate: 0.0007
```
- summary

| Type  | Pearson Correlation Coefficient | Root Mean Square Error |
| ----- | ------ | ------ |
| valid | 0.9423 | 0.1267 |
| test  | 0.9479 | 0.1386 |

# Citation
```
@INPROCEEDINGS{10890056,
  author={Park, Chanho and Lu, Chengsong and Chen, Mingjie and Hain, Thomas},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Fast Word Error Rate Estimation Using Self-Supervised Representations for Speech and Text}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Degradation;Correlation coefficient;Error analysis;Estimation;Bidirectional long short term memory;Self-supervised learning;Signal processing;Real-time systems;Speech processing;Root mean square;Word error rate;WER estimation;self-supervised representation;multi-layer perceptrons;inference speed},
  doi={10.1109/ICASSP49660.2025.10890056}}
```
