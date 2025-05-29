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
    activation              = args.activation
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
                                   train_label_file_full_path, rank)
    valid_dataset = SegmentDataset(valid_utterance_scp_file_full_path, \
                                   valid_hypothesis_scp_file_full_path, \
                                   valid_label_file_full_path, rank)

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
        for i, (stm_ids, utterance_samples, hypothesis_samples, wers, sub_rates, del_rates, ins_rates) in enumerate(train_loader):
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
            for i, (stm_ids, utterance_samples, hypothesis_samples, wers, sub_rates, del_rates, ins_rates) in enumerate(valid_loader):
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


if __name__ == "__main__":
    print('Initiating...')
    parser = argparse.ArgumentParser(prog='Model training', description='Trains a model for Word Error Rate (WER) estimation')
    parser.add_argument('--base_path', metavar='str', help='a file path to a base directory', required=True)
    parser.add_argument('--train_dataset_name', metavar='str', help='e.g., tl3_train', required=True)
    parser.add_argument('--valid_dataset_name', metavar='str', help='e.g., tl3_valid', required=True)
    parser.add_argument('--hypothesis_name', metavar='str', help='e.g., data.whisper_large', required=True)
    parser.add_argument('--utterance_encoder_name', metavar='str', help='an encoder name for an utterance', required=True)
    parser.add_argument('--transcript_encoder_name', metavar='str', help='an encoder name for a transcript', required=True)
    parser.add_argument('--batch_size', metavar='int', help='batch size for data load', required=True)
    parser.add_argument('--num_workers', metavar='int', help='the number of workers for data load', required=True)
    parser.add_argument('--max_duration', metavar='int', help='maximum length of utterance for data load', required=True)
    parser.add_argument('--model_path', metavar='str', help='a path to save a model', required=True)
    parser.add_argument('--layer_sizes', metavar='int', nargs='+', help='a layer number for MultiLayer Perceptrons', required=True)
    parser.add_argument('--dropout', metavar='float', help='dropout for training', required=True)
    parser.add_argument('--activation', metavar='str', help='a activation function name', required=True)
    parser.add_argument('--learning_rate', metavar='float', help='a learning rate for training', required=True)
    parser.add_argument('--max_iteration', metavar='int', help='the maximum iteration for scheduler', required=True)
    parser.add_argument('--max_epochs', metavar='int', help='the maximum epoch for training', required=True)
    parser.add_argument('--early_stop', metavar='int', help='the number of epochs for stopping training', required=True)

    args = parser.parse_args()
    print(args)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = torch.cuda.device_count()
    port = get_free_port()
    mp.spawn(training, args=(world_size, port, args), nprocs=world_size) 
