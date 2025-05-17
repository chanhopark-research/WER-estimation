import os
import glob
from pathlib import Path

import sys
sys.path.append('/share/mini1/res/t/asr/multi/multi-en/acsw/selft/repositories/whisper')
from whisper.normalizers import EnglishTextNormalizer
sys.path.append('/share/mini1/res/t/asr/multi/multi-en/acsw/selft/repositories/NeMo-text-processing')
import nemo_text_processing
from nemo_text_processing.text_normalization.normalize import Normalizer as NeMoNormalizer

import re

#==================================================================
# mapping dictionary
#==================================================================
mapping_dict = {
    r"\b401[kK]\b": "four o one k"
}

try:
    english_normalizer = EnglishTextNormalizer()
    nemo_normalizer = NeMoNormalizer(input_case='lower_cased', lang='en')
    print("Normalizers initialized successfully.")
except Exception as e:
    print(f"Error initializing normalizers: {e}")
    print("Please ensure 'whisper-normalizer' and 'nemo_text_processing' are correctly installed.")

#==================================================================
# text normalization
#==================================================================
def normalization(normalized):
    # special rules
    if dataset_name.startswith('swb'):
        normalized = normalized.replace('_1', '')
    # common rules
    normalized = re.sub(r"[<\[][^>\]]*[>\]]", "", normalized)  # remove words between brackets
    normalized = re.sub(r"\(([^)]+?)\)", "", normalized)  # remove words between parenthesis
    # customised mapping list
    for pattern, replacement in mapping_dict.items():
        normalized = re.sub(pattern, replacement, normalized)

    try:
        # NeMo normalizer e.g. AT&T -> AT and T, 125k -> one hundred twenty five k
        normalized = nemo_normalizer.normalize(normalized, punct_post_process=True)
    except:
        print(f'error occured: {normalized}')
    # whisper normalizer e.g. remove symbols, lower-case, won't -> will not, ...
    normalized = english_normalizer(normalized)
    normalized = normalized.split()
    if '.' in normalized:
        normalized.remove('.')
    normalized = ' '.join(normalized)
    return normalized

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
    output_audmap_file_full_path = f'{subset_path}/data.audmap'
    with open(output_stm_file_full_path, 'w') as output_stm_file, \
         open(output_audmap_file_full_path, 'w') as output_audmap_file:
        for stm_file in stm_files:
            # print(f'stm_file: {stm_file}')
            with open(stm_file, 'r') as input_stm_file:
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
                    print(f'{dataset_name.upper()}-{talk_id} {channels} {speaker_id} {start_time} {end_time} <id={segment_id},dset={subset_type}> {normalization(" ".join(transcript))}', file=output_stm_file)
                    print(f'{dataset_name.upper()}-{talk_id}_{segment_id} {channels} {start_time} {end_time} {audio_file_full_path}', file=output_audmap_file)
