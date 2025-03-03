import torchaudio
import torch

for subset_type in ['test', 'dev', 'train']:
    # Assuming you've already downloaded the dataset
    tedlium_dataset = torchaudio.datasets.TEDLIUM(
        root="/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/datasets",
        release="release3",
        subset=subset_type,
        download=False  # Set to False since we've already downloaded it
    )
    print(f'total setments: {len(tedlium_dataset)}')

    # Get a sample from the dataset
    sample_idx = 0
    sample = tedlium_dataset[sample_idx]
    
    # Each sample is a tuple containing:
    # (waveform, sample_rate, transcript, talk_id, speaker_id, identifier)
    waveform, sample_rate, transcript, talk_id, speaker_id, identifier = sample
    
    # Basic information about the sample
    print(f"Talk ID: {talk_id}")
    print(f"Speaker ID: {speaker_id}")
    print(f"Transcript: {transcript.strip()}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Waveform shape: {waveform.shape}")  # [channels, time]
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds\n")
