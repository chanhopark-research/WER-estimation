import argparse
import yaml
import logging
import torchaudio
from pathlib import Path
import os
import numpy as np
from kaldiio import WriteHelper
import torch
from torch.utils.data import Dataset, DataLoader
import fairseq
import sys

#==================================================================
# set logger
#==================================================================
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

#==================================================================
# set seed
#==================================================================
import random
def set_seed(seed=27):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#==================================================================
# build a segment list from stm and audmap files
#==================================================================
def get_id(meta_info):
    return meta_info.strip().split(',')[0][4:]

def build_segment_dict(dataset_name):
    # list of utterance_dict and transcripts
    utterance_dict = dict()
    transcript_dict = dict()

    dataset_path = f'/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets/{dataset_name}'
    audmap_file_full_path = f'{dataset_path}/data.audmap'
    stm_file_full_path = f'{dataset_path}/data.stm'

    # read an audmap file
    with open(audmap_file_full_path, 'r') as audmap_file:
        logging.info(f'audmap_file_full_path: {audmap_file_full_path}')
        for line in audmap_file:
            splited = line.strip().split()
            logging.debug(f'splited: {splited}')
            stm_id = splited[0]
            channel = int(splited[1]) - 1
            start_time = float(splited[2])
            end_time = float(splited[3])
            # remove utterances shorter than 0s
            if end_time < start_time:
                logging.info(f'{splited[0]} is shorter than 0')
                continue
            utterance_dict[stm_id] = dict()
            utterance_dict[stm_id]['channel'] = channel
            utterance_dict[stm_id]['start_time'] = start_time
            utterance_dict[stm_id]['end_time'] = end_time
            utterance_dict[stm_id]['full_path'] = splited[-1]

    # read a stm file
    with open(stm_file_full_path, 'r') as stm_file:
        logging.info(f'stm_file_full_path: {stm_file_full_path}')
        for line in stm_file:
            splited = line.strip().split()
            stm_id = f'{splited[0]}_{get_id(splited[5])}'
            logging.debug(f'stm_id: {stm_id}')
            transcript_dict[stm_id] = dict()
            transcript_dict[stm_id]['stm_info']   = (' ').join(splited[:6])
            transcript_dict[stm_id]['transcript'] = (' ').join(splited[6:])

    # common stm_ids
    stm_ids = set(utterance_dict.keys()) & set(transcript_dict.keys())
    return list(stm_ids), utterance_dict, transcript_dict


#=================================================================
# convert utterance to waveform
#==================================================================
def convert_utterance_to_waveform(utterance, sample_rate):
    channel    = utterance['channel']
    start_time = utterance['start_time']
    end_time   = utterance['end_time']
    full_path  = utterance['full_path']

    frame_offset = int(start_time * sample_rate)
    num_frames = int(end_time * sample_rate) - frame_offset

    if full_path.endswith('sph'): # switchboard
        waveform, sample_rate = torchaudio.backend.sox_io_backend.load(filepath=full_path, frame_offset=frame_offset, num_frames=num_frames, format='sph')
    elif full_path.endswith('flac'): # librispeech
        waveform, sample_rate = torchaudio.backend.sox_io_backend.load(filepath=full_path, frame_offset=frame_offset, num_frames=num_frames, format='flac')
    else:
        waveform, sample_rate = torchaudio.backend.sox_io_backend.load(filepath=full_path, frame_offset=frame_offset, num_frames=num_frames, format='wav')
    waveform = waveform[channel]

    assert sample_rate == sample_rate, 'sample_rate is different from sample_rate'

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform

#==================================================================
# split the list for parallel jobs
#==================================================================
def split_segment_for_parallel(stm_ids, total_jobs, job_number):
    total_lines = len(stm_ids)
    if total_jobs > 0:
        quotient = total_lines // total_jobs
        remainder = total_lines % total_jobs
        if remainder > 0:
            quotient += 1
        start_line = quotient * (job_number - 1)
        end_line = quotient * job_number
        if end_line > total_lines:
            end_line = total_lines
    else:
        start_line = 0
        end_line = total_lines
    logging.info(f"start_line: {start_line}, end_line: {end_line}, total_lines: {total_lines}")
    return start_line, end_line

#==================================================================
# save waveform embeddings - ark,scp
#==================================================================
def save_features_ark_scp(stm_ids, utterance_dict, transcript_dict, start_line, end_line, ark_file_name, scp_file_name, dataset_name, encoder_name, sample_rate, feature_encoder, feature_layer = int(-1), feature_level='sequence'):
    logging.info(f"save {encoder_name} features - ark,scp: {len(stm_ids)} segments")

    with WriteHelper(f'ark,scp:{ark_file_name},{scp_file_name}') as writer:
        for key in stm_ids[start_line:end_line]:
            try: 
                logging.info(f'key: {key}')
                # audio: hubert
                if encoder_name.startswith('hubert'):
                    waveform = convert_utterance_to_waveform(utterance_dict[key], sample_rate)
                    features, _ = feature_encoder.extract_features(waveform[None, :].to(device)) # [layer][batch_size, frames, 1024]
                    features = features[feature_layer - 1]
                # text: xlmr
                elif encoder_name.startswith('xlmr'):
                    tokens = feature_encoder.encode(transcript_dict[key]['transcript'])
                    features = feature_encoder.extract_features(tokens.to(device))
                else:
                    logging.error('ERROR: unknown encoder: {encoder_name}')
                # aggregate raw features
                if feature_level == 'sequence':
                    features = torch.mean(features, dim=1)  # (1, 1024)
                else:
                    features = features[0]
                logging.info(f'features.shape: {features.shape}')
                # save features
                writer(key, features.cpu().detach().numpy())
                # cleanup
                del features
                torch.cuda.empty_cache()

            except RuntimeError as e:
                logging.error(f"Runtime error for key {key}: {e}")
                torch.cuda.empty_cache()

import argparse

if __name__ == "__main__":
    print('Initiating...')
    parser = argparse.ArgumentParser(prog='Feature extraction', description='Extract features for WER estimation')
    parser.add_argument('--total_jobs', metavar='int', help='the number of parallel jobs', required=True)
    parser.add_argument('--job_number', metavar='int', help='the job number', required=True)
    parser.add_argument('--dataset_name', metavar='str', help='Dataset name', required=True)
    parser.add_argument('--feature_name', metavar='str', help='audio: hubert, text: xlmr', required=True)
    parser.add_argument('--training_type', metavar='str', help='ft for fine-tuned, pt for pre-trained', required=True)
    parser.add_argument('--sample_rate', metavar='int', help='sampling rate', required=True)
    parser.add_argument('--feature_layer', metavar='int', help='the layer number', required=True)
    parser.add_argument('--feature_level', metavar='str', help='sequence for sequence-level, othewise frame-level', required=True)
    parser.add_argument('--feature_size', metavar='str', help='additional information for a model size', required=True)

    args = parser.parse_args()
    total_jobs    = int(args.total_jobs)
    job_number    = int(args.job_number)
    dataset_name  = args.dataset_name  
    feature_name  = args.feature_name  
    training_type = args.training_type 
    sample_rate   = int(args.sample_rate)
    feature_layer = int(args.feature_layer)
    feature_level = args.feature_level 
    feature_size  = args.feature_size  
    print(args)
    encoder_name = f'{feature_name}-{training_type}-{feature_layer}-{feature_level}.{feature_size}'

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stm_ids, utterance_dict, transcript_dict = build_segment_dict(dataset_name)
    stm_ids.sort()
    start_line, end_line = split_segment_for_parallel(stm_ids)

    output_dir = f"/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/features/{dataset_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ark_file_name = f"{output_dir}/{encoder_name}_{total_jobs}_{job_number}.ark"
    scp_file_name = f"{output_dir}/{encoder_name}_{total_jobs}_{job_number}.scp"

    #==================================================================
    # load models
    #==================================================================
    # audio: hubert
    if feature_name == 'hubert' and training_type == 'pt' and feature_size == 'large':
        bundle = torchaudio.pipelines.HUBERT_LARGE
        feature_encoder = bundle.get_model()
    elif feature_name == 'hubert' and training_type == 'ft' and feature_size == 'large':
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        feature_encoder = bundle.get_model()
    # text: xlmr
    elif feature_name == 'xlmr' and training_type == 'pt' and feature_size == 'large':
        feature_encoder = torch.hub.load('pytorch/fairseq', 'xlmr.large')
    feature_encoder.to(device).eval()

    logging.info(f'feature_encoder: {feature_encoder}')
    save_features_ark_scp(stm_ids, utterance_dict, transcript_dict, start_line, end_line, ark_file_name, scp_file_name, dataset_name, encoder_name, sample_rate, feature_encoder, feature_layer, feature_level)
