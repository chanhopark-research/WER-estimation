import os
import glob
from pathlib import Path

dataset_path = '/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets'
dataset_name = 'tl3'
for subset_type in ['train', 'dev', 'test']:
    # read from
    audio_path = f'{dataset_path}/TEDLIUM_release-3/legacy/{subset_type}/sph'
    stm_path = f'{dataset_path}/TEDLIUM_release-3/legacy/{subset_type}/stm'
    stm_files = glob.glob(os.path.join(stm_path, '*.stm'))

    # write to
    subset_path = f'{dataset_path}/{dataset_name}_{subset_type}'
    Path(subset_path).mkdir(parents=True, exist_ok=True)
    output_stm_file_full_path = f'{subset_path}/data.stm'
    output_audmap_file_full_path = f'{subset_path}/audmap.stm'

    for stm_file in stm_files:
        # print(f'stm_file: {stm_file}')
        with open(stm_file, 'r') as input_stm_file, \
             open(output_stm_file_full_path, 'w') as output_stm_file, \
             open(output_audmap_file_full_path, 'w') as output_audmap_file:
            for line in input_stm_file:
                parsed = line.strip().split()
                talk_id = parsed[0] 
                channels = parsed[1]
                speaker_id = parsed[2]
                start_time = parsed[3]
                end_time = parsed[4]
                meta_info = parsed[5]
                transcript = parsed[6:] 
                segment_id = f'{channels}:{start_time.replace(".", "")}-{end_time.replace(".", "")}'
                audio_file_full_path = os.path.join(audio_path, f'{talk_id}.sph')
                print(f'{talk_id}_{segment_id} {channels} {start_time} {end_time} {audio_file_full_path}', file=output_audmap_file)
                print(f'{talk_id} {channels} {speaker_id} {start_time} {end_time} <id={segment_id},dset={subset_type}> {" ".join(transcript)}', file=output_stm_file)
