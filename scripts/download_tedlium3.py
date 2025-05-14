import torchaudio

tedlium_dataset = torchaudio.datasets.TEDLIUM(
    root=f"/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets"
    release="release3",  # Options: "release1", "release2", or "release3"
    subset=None,         # None for downloading
    download=True        # Set to True to download the dataset
)
