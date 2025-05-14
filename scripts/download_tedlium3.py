import torchaudio

tedlium_dataset = torchaudio.datasets.TEDLIUM(
    root="/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets",  # Specify the directory where you want to store the dataset
    release="release3",  # You can choose "release1", "release2", or "release3"
    subset=None,         # None for download only
    download=True        # Set to True to download the dataset
)
