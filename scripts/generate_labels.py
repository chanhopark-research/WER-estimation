from functions import *
import argparse

if __name__ == "__main__":
    print('Initiating...')
    parser = argparse.ArgumentParser(prog='Label generation', description='Compute Word Error Rate (WER)')
    parser.add_argument('--dataset_name', metavar='str', help='Dataset name', required=True)
    parser.add_argument('--model_name', metavar='str', help='ASR model name, e.g., whisper', required=True)
    parser.add_argument('--model_size', metavar='str', help='additional information for a model size', required=True)

    args = parser.parse_args()
    dataset_name  = args.dataset_name  
    model_name    = args.model_name
    model_size    = args.model_size  
    print(args)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_path = f'/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets/{dataset_name}'
    reference_path = f'/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/datasets/{dataset_name}'
    hypothesis_path = f'/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/hypotheses/{dataset_name}'

    utterance_dict = read_audmap_file(f'{dataset_path}/data.audmap', duration_only=True)
    reference_dict = read_stm_file(f'{reference_path}/data.stm', transcript_only=True)
    hypothesis_dict = read_stm_file(f'{hypothesis_path}/data.{model_name}_{model_size}.stm', transcript_only=True)

    # common stm_ids
    stm_ids = set(utterance_dict.keys()) & set(reference_dict.keys()) & set(hypothesis_dict.keys())

    label_path = f'/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/labels/{dataset_name}'
    Path(label_path).mkdir(parents=True, exist_ok=True)
    label_file_full_path = f'{label_path}/data.{model_name}_{model_size}.wer'

    save_labels(stm_ids, utterance_dict, reference_dict, hypothesis_dict, label_file_full_path)
