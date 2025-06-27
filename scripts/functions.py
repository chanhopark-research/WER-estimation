from classes import *
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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import fairseq
import random
import glob
import sys
sys.path.append('/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/repositories/whisper')
import whisper
from whisper.normalizers import EnglishTextNormalizer
sys.path.append('/share/mini1/res/t/asr/multi/multi-en/acsw/selft/repositories/NeMo-text-processing')
import nemo_text_processing
from nemo_text_processing.text_normalization.normalize import Normalizer as NeMoNormalizer
from speechbrain.utils.edit_distance import wer_details_by_utterance
import re
import time
import scipy.stats
from sklearn.metrics import mean_squared_error

#==================================================================
# global variables
#==================================================================
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

#==================================================================
# set seed
#==================================================================
def set_seed(seed=27):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#==================================================================
# set up ddp environment
#==================================================================
def is_main_process(rank: int):
    return rank == 0

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f'rank: {rank}, world_size: {world_size}, str(port): {str(port)}')

#==================================================================
# build a segment list from stm and audmap files
#==================================================================
def get_id(meta_info):
    return meta_info.strip().split(',')[0][4:]

def read_audmap_file(audmap_file_full_path, duration_only=False):
    utterance_dict = dict()
    # read an audmap file
    with open(audmap_file_full_path, 'r') as audmap_file:
        logging.info(f'audmap_file_full_path: {audmap_file_full_path}')
        for line in audmap_file:
            parts = line.strip().split()
            logging.debug(f'parts: {parts}')
            stm_id = parts[0]
            channel = int(parts[1]) - 1
            start_time = float(parts[2])
            end_time = float(parts[3])
            # remove utterances shorter than 0s
            if end_time < start_time:
                logging.info(f'{parts[0]} is shorter than 0')
                continue
            if duration_only:
                utterance_dict[stm_id] = float(end_time - start_time)
            else:
                utterance_dict[stm_id] = dict()
                utterance_dict[stm_id]['channel'] = channel
                utterance_dict[stm_id]['start_time'] = start_time
                utterance_dict[stm_id]['end_time'] = end_time
                utterance_dict[stm_id]['full_path'] = parts[-1]
    return utterance_dict

def read_stm_file(stm_file_full_path, transcript_only=False):
    transcript_dict = dict()
    # read a stm file
    with open(stm_file_full_path, 'r') as stm_file:
        logging.info(f'stm_file_full_path: {stm_file_full_path}')
        for line in stm_file:
            parts = line.strip().split()
            stm_id = f'{parts[0]}_{get_id(parts[5])}'
            logging.debug(f'stm_id: {stm_id}')
            if transcript_only:
                transcript_dict[stm_id] = parts[6:]
            else:
                transcript_dict[stm_id] = dict()
                transcript_dict[stm_id]['stm_info']   = (' ').join(parts[:6])
                transcript_dict[stm_id]['transcript'] = (' ').join(parts[6:])
    return transcript_dict

def build_segment_dict(dataset_name):
    dataset_path = f'/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets/{dataset_name}'
    audmap_file_full_path = f'{dataset_path}/data.audmap'
    stm_file_full_path = f'{dataset_path}/data.stm'

    utterance_dict = read_audmap_file(audmap_file_full_path)
    transcript_dict = read_stm_file(stm_file_full_path)

    # common stm_ids
    stm_ids = set(utterance_dict.keys()) & set(transcript_dict.keys())
    return list(stm_ids), utterance_dict, transcript_dict

#==================================================================
# read features: e.g. hubert and roberta
#==================================================================
def read_scp_file(scp_file_name, keep_dim=False):
    logger.debug(f'scp_file_name: {scp_file_name}')
    start_time = time.time()
    mat_dict = dict()
    with ReadHelper(f'scp:{scp_file_name}') as reader:
        for key, mat in reader:
            if keep_dim:
                mat_dict[key] = mat  # [frames, 1024]
            else:
                mat_dict[key] = mat[0]  # [1, 1024] -> [1024]
    logger.debug(f'time on reading {len(mat_dict)} data : {time.time() - start_time:.4f}')
    return mat_dict

def read_features(dataset_name, utterance_encoder_name, transcript_encoder_name):
    utterance_encoder_file_name = f'{features_path}/{dataset_name}/{utterance_encoder_name}.scp'
    transcript_encoder_file_name = f'{features_path}/{dataset_name}/{transcript_encoder_name}.scp'
    logger.info(f'utterance_encoder_file_name: {utterance_encoder_file_name}')
    logger.info(f'transcript_encoder_file_name: {transcript_encoder_file_name}')
    utterance_encoder_dict = read_scp_file(utterance_encoder_file_name)
    utterance_stm_id_set = set(list(utterance_encoder_dict.keys()))
    transcript_encoder_dict = read_scp_file(transcript_encoder_file_name)
    transcript_stm_id_set = set(list(transcript_encoder_dict.keys()))
    stm_ids_difference = utterance_stm_id_set ^ transcript_stm_id_set # symmetric difference = union - intersection
    for stm_id in stm_ids_difference:
        if stm_id in utterance_encoder_dict:
            del(utterance_encoder_dict[stm_id])
        if stm_id in transcript_encoder_dict:
            del(transcript_encoder_dict[stm_id])
    if set(utterance_encoder_dict.keys()) != set(transcript_encoder_dict.keys()):
        sys.exit('ERROR: the number of segments are not the same')
    return utterance_encoder_dict.keys(), utterance_encoder_dict, transcript_encoder_dict

def read_feature_file(file_full_path, rank, keep_dim: bool = False):
    from kaldiio import ReadHelper
    feature_dict = dict()
    import time
    start_time = time.time()
    with ReadHelper(f'scp:{file_full_path}') as reader:
        #print(f'file_full_path: {file_full_path}')
        for key, mat in reader:
            #print(f'mat: {mat.shape}')
            if keep_dim:
                feature_dict[key] = mat  # [frames, 1024]
            else:
                feature_dict[key] = mat[0]  # [1, 1024] -> [1024]
    if is_main_process(rank):
        logger.debug(f'read features: {file_full_path}, time: {time.time() - start_time:.4f} seconds')
    return feature_dict

def read_label_file(file_full_path, stm_id_list, max_duration, rank, verbose=True):
    label_dict = dict()
    with open(file_full_path) as label_file:
        for line in label_file:
            parts = line.strip().split(',')
            stm_id = parts[0]
            if stm_id not in stm_id_list:
                if verbose:
                    logger.debug(f'not in features: skipped {stm_id}')
                continue
            import numpy as np
            tkn_er = float(np.clip(float(parts[1]) / 100, a_min=0.0, a_max=1.0))
            num_sub = int(parts[2])
            num_del = int(parts[3])
            num_ins = int(parts[4])
            num_edi = int(parts[5])
            num_ref = int(parts[6])
            duration = float(parts[7])
            if duration <= max_duration:
                continue
            if stm_id not in label_dict:
                label_dict[stm_id] = dict()
            label_dict[stm_id]['tkn_er'] = tkn_er
            label_dict[stm_id]['num_sub'] = num_sub
            label_dict[stm_id]['num_del'] = num_del
            label_dict[stm_id]['num_ins'] = num_ins
            label_dict[stm_id]['num_edi'] = num_edi
            label_dict[stm_id]['num_ref'] = num_ref
            label_dict[stm_id]['duration'] = duration
    if is_main_process(rank):
        logger.debug(f'read labels: {file_full_path}')
    return label_dict

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

#==================================================================
# text normalization
#==================================================================
def normalization(transcript, mapping_dict, english_normalizer, nemo_normalizer):
    # common rules
    transcript = re.sub(r"[<\[][^>\]]*[>\]]", "", transcript)  # remove words between brackets
    transcript = re.sub(r"\(([^)]+?)\)", "", transcript)  # remove words between parenthesis
    # customised mapping list
    for pattern, replacement in mapping_dict.items():
        transcript = re.sub(pattern, replacement, transcript)

    try:
        # NeMo normalizer e.g. AT&T -> AT and T, 125k -> one hundred twenty five k
        transcript = nemo_normalizer.normalize(transcript, punct_post_process=True)
    except:
        print(f'error occured: {transcript}')
    # whisper normalizer e.g. remove symbols, lower-case, won't -> will not, ...
    transcript = english_normalizer(transcript)
    transcript = transcript.split()
    if '.' in transcript:
        transcript.remove('.')
    if '..' in transcript:
        transcript.remove('..')
    transcript = ' '.join(transcript)
    return transcript

#===================================================
# save hypotheses generated by whisper
#===================================================
def save_hypotheses(model_name: str, model_size: str, dataset_name: str, sample_rate: int, stm_ids: list, utterance_dict: dict, transcript_dict: dict, start_line: int, end_line: int, hypothesis_file_name: str, job_number: int, total_jobs: int):

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
    # load models
    #==================================================================
    model = whisper.load_model(model_size) # available models = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']
    logging.info(
        f"Model is {model_name} {model_size}, {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    with open(hypothesis_file_name, 'w') as output_file:
        for stm_id in stm_ids[start_line:end_line]:
            logging.info(f'stm_id: {stm_id}')
            waveform = convert_utterance_to_waveform(utterance_dict[stm_id], sample_rate)
            result = model.transcribe(waveform)
            hypothesis = result['text']
            print(f'{transcript_dict[stm_id]["stm_info"]} {normalization(hypothesis, mapping_dict, english_normalizer, nemo_normalizer)}', file=output_file)

#==================================================================
# compute WER
#==================================================================
def save_labels(stm_ids, utterance_dict, reference_dict, hypothesis_dict, label_file_full_path, compute_alignments=True, scoring_mode='all', duration_limit=float('inf')):
    try:
        wer_details = wer_details_by_utterance(reference_dict, hypothesis_dict, compute_alignments=compute_alignments, scoring_mode=scoring_mode)
    except:
        for key in reference_dict:
            print(f'exception: reference_dict[{key}]: {reference_dict[key]}, hypothesis_dict[{key}]: {hypothesis_dict[key]}')

    total_seg = 0
    total_tok = 0
    total_del = 0
    total_sub = 0
    total_ins = 0
    total_dur = 0
    total_WER = 0
    WER_list = list()
    with open(label_file_full_path, 'w') as output_file:
        logging.info('label_file_full_path: {label_file_full_path}')
        for detail in wer_details:
            #logging.info(detail)
            if detail['hyp_absent'] and detail['key'] not in stm_ids:
                continue
            stm_id = detail['key']
            WER = float(detail['WER'])
            substitutions = detail['substitutions']
            deletions = detail['deletions']
            insertions = detail['insertions']
            num_edits = detail['num_edits']
            num_ref_tokens = detail['num_ref_tokens']
            duration = float(utterance_dict[stm_id])
            print(f'{stm_id},{WER},{substitutions},{deletions},{insertions},{num_edits},{num_ref_tokens},{duration}', file=output_file)
            if duration <= float(duration_limit):
                total_seg += 1
                total_WER += WER/100
                total_sub += substitutions
                total_del += deletions
                total_ins += insertions
                total_tok += num_ref_tokens
                total_dur += duration
                WER_list.append(WER/100)
    print(f'total_seg: {total_seg}, total_dur(h): {total_dur/3600:.2f}, avg_dur(s): {total_dur / total_seg:.2f}, total_tok: {total_tok}, avg_tok: {total_tok / total_seg:.4f}, avg_WER: {np.mean(WER_list):.4f}, std. dev. WER: {np.std(WER_list):.4f}, WER (weighted): {(total_ins + total_del + total_sub) / total_tok:.4f}, total_edits: {total_ins + total_del + total_sub}, total_ins: {total_ins}, total_del: {total_del}, total_sub: {total_sub}')

def get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost',0),None) as s:
           return s.server_address[1]

def training(rank: int, world_size: int, port: int, args: dict()):
    ddp_setup(rank, world_size, port)
    set_seed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    base_path               = args.base_path
    train_dataset_name      = args.train_dataset_name
    valid_dataset_name      = args.valid_dataset_name
    utterance_encoder_name  = args.utterance_encoder_name
    transcript_encoder_name = args.transcript_encoder_name
    hypothesis_name         = args.hypothesis_name
    batch_size              = int(args.batch_size)
    num_workers             = int(args.num_workers)
    max_duration            = int(args.max_duration)
    model_path              = args.model_path
    layer_sizes             = [int(layer_size) for layer_size in list(args.layer_sizes)]
    dropout                 = float(args.dropout)
    learning_rate           = float(args.learning_rate)
    max_iteration           = int(args.max_iteration)
    max_epochs              = int(args.max_epochs)
    early_stop              = int(args.early_stop)

    #==================================================================
    # load features
    #==================================================================
    train_utterance_scp_file_full_path  = f'{base_path}/features/{train_dataset_name}/{utterance_encoder_name}.scp'
    valid_utterance_scp_file_full_path  = f'{base_path}/features/{valid_dataset_name}/{utterance_encoder_name}.scp'
    train_hypothesis_scp_file_full_path = f'{base_path}/features/{train_dataset_name}/{transcript_encoder_name}.scp'
    valid_hypothesis_scp_file_full_path = f'{base_path}/features/{valid_dataset_name}/{transcript_encoder_name}.scp'
    train_label_file_full_path          = f'{base_path}/labels/{train_dataset_name}/data.{hypothesis_name}.wer'
    valid_label_file_full_path          = f'{base_path}/labels/{valid_dataset_name}/data.{hypothesis_name}.wer'

    train_dataset = SegmentDataset(train_utterance_scp_file_full_path, \
                                   train_hypothesis_scp_file_full_path, \
                                   train_label_file_full_path, max_duration, rank)
    valid_dataset = SegmentDataset(valid_utterance_scp_file_full_path, \
                                   valid_hypothesis_scp_file_full_path, \
                                   valid_label_file_full_path, max_duration, rank)

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(train_dataset))
    valid_loader = DataLoader(dataset = valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_total_steps = len(train_loader)
    valid_total_steps = len(valid_loader)

    if is_main_process(rank):
        logger.info(f'train_total_steps: {train_total_steps}')
        logger.info(f'valid_total_steps: {valid_total_steps}')

    #==================================================================
    # load models
    #==================================================================
    model = MultipleHiddenLayersModel(layer_sizes, dropout)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iteration, verbose=False)

    os.makedirs(model_path, exist_ok=True)
    best_model_full_path = f'{model_path}/best.pt'
    last_model_full_path = f'{model_path}/last.pt'

    start_epoch = 0
    best_valid_loss_epoch = 0
    not_improved_count = 0
    if is_main_process(rank):
        logger.info(f'rank: {rank} | A new mapping model has been generated.')

    #==================================================================
    # WER prediction
    #==================================================================
    loss = nn.MSELoss(reduction='mean')
    for current_epoch in range(start_epoch, max_epochs):
        train_loss_epoch = 0
        valid_loss_epoch = 0

        #==============================================================
        # training
        #==============================================================
        start_time_epoch = time.time()
        model.module.train()
        for i, (stm_ids, utterance_samples, hypothesis_samples, sub_rates, ins_rates, del_rates, wers) in enumerate(train_loader):
            start_time_step = time.time()
            # Loss
            logit_wer = model(utterance_samples.to(device), hypothesis_samples.to(device))
            train_loss_step = loss(logit_wer, wers.to(device)[:,None])
            train_loss_epoch += float(train_loss_step)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss_step.backward()
            optimizer.step()

            # step logging
            end_time_step = time.time()

        #==============================================================
        # validation
        #==============================================================
        predicted_wer_list = list()
        reference_wer_list = list()

        model.module.eval()
        with torch.no_grad():
            for i, (stm_ids, utterance_samples, hypothesis_samples, sub_rates, ins_rates, del_rates, wers) in enumerate(valid_loader):
                logit_wer = model(utterance_samples.to(device), hypothesis_samples.to(device))
                predicted_wer_list += torch.squeeze(logit_wer)
                reference_wer_list += wers

                # loss
                valid_loss_step = loss(logit_wer, wers.to(device)[:,None])
                valid_loss_epoch += float(valid_loss_step)

        predicted_wer_numpy_list = list()
        reference_wer_numpy_list = list()
        for t in predicted_wer_list:
            predicted_wer_numpy_list.append(t.detach().cpu().numpy())
        for t in reference_wer_list:
            reference_wer_numpy_list.append(t.detach().cpu().numpy())
        predicted_wer_numpy_list = np.array(predicted_wer_numpy_list)
        reference_wer_numpy_list = np.array(reference_wer_numpy_list)

        pearson_correlation_coefficients = scipy.stats.pearsonr(predicted_wer_numpy_list, reference_wer_numpy_list)
        rmse = mean_squared_error(reference_wer_numpy_list, predicted_wer_numpy_list, squared=False)

        #==============================================================
        # scheduling
        #==============================================================
        # Backward and optimize
        scheduler.step()

        #==============================================================
        # epoch logging
        #==============================================================
        train_loss_epoch = torch.sqrt(torch.tensor(train_loss_epoch) / train_total_steps)
        valid_loss_epoch = torch.sqrt(torch.tensor(valid_loss_epoch) / valid_total_steps)

        end_time_epoch = time.time()
        if is_main_process(rank):
            logger.info(f'rank: {rank} | epoch: {current_epoch:3}, time: {end_time_epoch - start_time_epoch:5.2f}, train_loss_epoch: {train_loss_epoch:5.4f}, valid_loss_epoch: {valid_loss_epoch:5.4f}, valid_PCC: {pearson_correlation_coefficients[0]:5.4f}, valid_RMSE: {rmse:5.4f}')

        #==============================================================
        # save mapping models
        #==============================================================
        save_dict = dict()
        save_dict['epoch'] = current_epoch
        save_dict[f'model'] = model.module.state_dict()
        save_dict[f'optimizer'] = optimizer.state_dict()
        save_dict[f'scheduler'] = scheduler.state_dict()

        if current_epoch == 0 or best_valid_loss_epoch >= valid_loss_epoch: # if the current loss is better than the best
            best_valid_loss_epoch = valid_loss_epoch
            not_improved_count = 0
            save_dict['best_loss'] = best_valid_loss_epoch
            save_dict['not_improved_count'] = not_improved_count
            if is_main_process(rank):
                torch.save(save_dict, best_model_full_path)
                logger.info(f'rank: {rank} | The model is saved at {current_epoch} epoch.')
        else:
            not_improved_count += 1
            save_dict['best_loss'] = best_valid_loss_epoch
            save_dict['not_improved_count'] = not_improved_count
            if is_main_process(rank):
                logger.info(f'rank: {rank} | not_improved_count/early_stop: {not_improved_count}/{early_stop}')
        if is_main_process(rank):
            torch.save(save_dict, last_model_full_path)

        if not_improved_count >= early_stop:
            if is_main_process(rank):
                logger.info(f'rank: {rank} | Early stop at {current_epoch} epoch after {not_improved_count} epochs.')
            break

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

def evaluation(rank: int, world_size: int, port: int, args: dict()):
    ddp_setup(rank, world_size, port)
    set_seed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    base_path               = args.base_path
    test_dataset_name       = args.test_dataset_name
    hypothesis_name         = args.hypothesis_name
    utterance_encoder_name  = args.utterance_encoder_name
    transcript_encoder_name = args.transcript_encoder_name
    batch_size              = int(args.batch_size)
    num_workers             = int(args.num_workers)
    max_duration            = int(args.max_duration)
    model_path              = args.model_path
    layer_sizes             = [int(layer_size) for layer_size in list(args.layer_sizes)]
    exp_path                = args.exp_path

    #==================================================================
    # load features
    #==================================================================
    test_utterance_scp_file_full_path  = f'{base_path}/features/{test_dataset_name}/{utterance_encoder_name}.scp'
    test_hypothesis_scp_file_full_path = f'{base_path}/features/{test_dataset_name}/{transcript_encoder_name}.scp'
    test_label_file_full_path          = f'{base_path}/labels/{test_dataset_name}/data.{hypothesis_name}.wer'

    test_dataset = SegmentDataset(test_utterance_scp_file_full_path, \
                                   test_hypothesis_scp_file_full_path, \
                                   test_label_file_full_path, max_duration, rank)

    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(test_dataset))

    test_total_steps = len(test_loader)

    if is_main_process(rank):
        logger.info(f'test_total_steps: {test_total_steps}')

    #==================================================================
    # load models
    #==================================================================
    model = MultipleHiddenLayersModel(layer_sizes)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    model_full_path = f'{model_path}/best.pt'
    if model_full_path and Path(model_full_path).is_file():
        checkpoint = torch.load(model_full_path, map_location=device)
        model.module.load_state_dict(checkpoint[f'model'])
        start_epoch = checkpoint['epoch'] + 1
        best_dev_loss_epoch = checkpoint['best_loss']
        not_improved_count = checkpoint['not_improved_count']
        if is_main_process(rank):
            logger.info(f'rank: {rank} | {model_full_path} has been loaded.')
    else:
        sys.exit(f'Model path is not valid: {model_full_path}')

    #==================================================================
    # WER prediction
    #==================================================================
    loss = nn.MSELoss(reduction='mean')
    test_loss_epoch = 0

    #==============================================================
    # evaluation
    #==============================================================
    predicted_wer_list = list()
    reference_wer_list = list()

    model.module.eval()
    reference_wers_torch = torch.empty((0))
    predicted_wers_torch = torch.empty((0))
    stm_id_list = list()
    with torch.no_grad():
        for i, (stm_ids, utterance_samples, hypothesis_samples, wers, sub_rates, del_rates, ins_rates) in enumerate(test_loader):
            # loss
            reference_wers_torch = torch.cat((reference_wers_torch.to(device), wers.to(device)))
            logit_wer = model(utterance_samples.to(device), hypothesis_samples.to(device))
            test_loss_step = loss(logit_wer, wers.to(device)[:,None])
            predicted_wers_torch = torch.cat((predicted_wers_torch.to(device), torch.squeeze(logit_wer).to(device)))
            test_loss_epoch += float(test_loss_step)
            predicted_wer_list += torch.squeeze(logit_wer)
            reference_wer_list += wers
            # for csv
            stm_id_list += stm_ids

    #csv: stm_id WER_reference WER_estimated duration
    ref_est_WER_output_file_name = f'{exp_path}/{test_dataset_name}_reference_and_estimated.csv'
    with open(ref_est_WER_output_file_name, 'w') as ref_est_WER_output_file:
        for stm_id, reference_wer, predicted_wer in zip(stm_id_list, reference_wers_torch, predicted_wers_torch):
            print(f'{stm_id} {float(reference_wer)} {float(predicted_wer)}', file=ref_est_WER_output_file)

    predicted_wer_numpy_list = list()
    reference_wer_numpy_list = list()
    for t in predicted_wer_list:
        predicted_wer_numpy_list.append(t.detach().cpu().numpy())
    for t in reference_wer_list:
        reference_wer_numpy_list.append(t.detach().cpu().numpy())
    predicted_wer_numpy_list = np.array(predicted_wer_numpy_list)
    reference_wer_numpy_list = np.array(reference_wer_numpy_list)

    pearson_correlation_coefficients = scipy.stats.pearsonr(predicted_wer_numpy_list, reference_wer_numpy_list)
    rmse = mean_squared_error(reference_wer_numpy_list, predicted_wer_numpy_list, squared=False)

    #==============================================================
    # epoch logging
    #==============================================================
    test_loss_epoch = torch.sqrt(torch.tensor(test_loss_epoch) / test_total_steps)

    end_time_epoch = time.time()
    if is_main_process(rank):
        logger.info(f'rank: {rank} | epoch: {start_epoch-1:3}, test_loss_epoch: {test_loss_epoch:5.4f}, test_PCC: {pearson_correlation_coefficients[0]:5.4f}, test_RMSE: {rmse:5.4f}')

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
