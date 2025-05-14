import torchaudio
import torch
import os

# Define constants
ROOT = "/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets"
RELEASE = "release3"

for subset_type in ["train", "dev", "test"]:
    # Assuming you've already downloaded the dataset
    tedlium_dataset = torchaudio.datasets.TEDLIUM(
        root=ROOT,
        release=RELEASE,
        subset=subset_type,
        download=False  # Set to False since we've already downloaded it
    )
    print(f'Total segments in {subset_type}: {len(tedlium_dataset)}')

    # Get a sample from the dataset
    sample_idx = 0
    sample = tedlium_dataset[sample_idx]
    print(f'sample: {sample}')

    waveform, sample_rate, transcript, talk_id, speaker_id, identifier = sample

    # Reconstruct the expected path to the .sph audio file
    # Release3 typically stores audio in: {ROOT}/TEDLIUM_release-3/data/{subset}/{talk_id}/{identifier}.sph
    audio_path = os.path.join(
        ROOT,
        f"TEDLIUM_release-3",  # hardcoded release name
        "legacy",
        subset_type,
        "sph",
        f"{talk_id}.sph"
    )

    # Basic information about the sample
    print(f"\nSubset: {subset_type}")
    print(f"Talk ID: {talk_id}")
    print(f"Speaker ID: {speaker_id}")
    print(f"Transcript: {transcript.strip()}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
    print(f"Audio file path: {audio_path}")
    print(f"Exists: {os.path.isfile(audio_path)}\n")

