from functions import *
from classes import *
import argparse
import time

import torch
import scipy.stats
from sklearn.metrics import mean_squared_error

def get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost',0),None) as s:
           return s.server_address[1]

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
                                   test_label_file_full_path, rank)

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


if __name__ == "__main__":
    print('Initiating...')
    parser = argparse.ArgumentParser(prog='Model evaluation', description='Evaluates a model for Word Error Rate (WER) estimation')
    parser.add_argument('--base_path', metavar='str', help='a file path to a base directory', required=True)
    parser.add_argument('--test_dataset_name', metavar='str', help='e.g., tl3_train', required=True)
    parser.add_argument('--hypothesis_name', metavar='str', help='e.g., data.whisper_large', required=True)
    parser.add_argument('--utterance_encoder_name', metavar='str', help='an encoder name for an utterance', required=True)
    parser.add_argument('--transcript_encoder_name', metavar='str', help='an encoder name for a transcript', required=True)
    parser.add_argument('--batch_size', metavar='int', help='batch size for data load', required=True)
    parser.add_argument('--num_workers', metavar='int', help='the number of workers for data load', required=True)
    parser.add_argument('--max_duration', metavar='int', help='maximum length of utterance for data load', required=True)
    parser.add_argument('--model_path', metavar='str', help='a path to save a model', required=True)
    parser.add_argument('--layer_sizes', metavar='int', nargs='+', help='a layer number for MultiLayer Perceptrons', required=True)
    parser.add_argument('--exp_path', metavar='str', help='a file path for saving the inference result ', required=True)

    args = parser.parse_args()
    print(args)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = torch.cuda.device_count()
    port = get_free_port()
    mp.spawn(evaluation, args=(world_size, port, args), nprocs=world_size) 
