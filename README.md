# WER-estimation
Implementation of Error Rate estimation models from research papers

# Task List
- [x] Download TEDLIUM3 using torchaudio
- [ ] Feature extraction (HuBERT, XLM-R)
- [ ] Transcribe speech using Automatic Speech Recognitioni (ASR) models (Whisper/HuBERT)
- [ ] Generate labels for Character Error Rate (CER) and Word Error Rate (WER)
- [ ] Train a CER/WER estimation model
- [ ] Evaluate the model

# datasets
- dev: 591 segments/1.71 hours
- test: 1469 segments/3.05 hours
- train: 268263 segments/453.81 hours

# normalizer
```
/share/mini1/res/t/asr/multi/multi-en/acsw/selft/repositories/whisper/whisper/normalizers/english.py
#s = self.standardize_numbers(s)
```

# results
- training_Fe-WER_600_32_0.0007/training_Fe-WER_600_32_0.0007.log
```
epoch:  41, train_loss_epoch: 0.0816, valid_loss_epoch: 0.1221, valid_PCC: 0.9423, valid_RMSE: 0.1267
epoch:  41, test_loss_epoch: 0.1352, test_PCC: 0.9479, test_RMSE: 0.1386
```
