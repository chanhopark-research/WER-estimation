from functions import *
from classes import *
import argparse
import torch

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
