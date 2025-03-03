import torchaudio

# Download TEDLium dataset (this may take some time as it's a large dataset)
tedlium_dataset = torchaudio.datasets.TEDLIUM(
    root="/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/datasets",  # Specify the directory where you want to store the dataset
    release="release3",  # You can choose "release1", "release2", or "release3"
    subset="test",  # Options: "train", "dev", "test"
    download=True  # Set to True to download the dataset
)
