from functions import *
from classes import *
import argparse
import torch

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
