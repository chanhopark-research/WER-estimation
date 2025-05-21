from utils import *
import argparse

if __name__ == "__main__":
    print('Initiating...')
    parser = argparse.ArgumentParser(prog='Feature extraction', description='Extract features for WER estimation')
    parser.add_argument('--total_jobs', metavar='int', help='the number of parallel jobs', required=True)
    parser.add_argument('--job_number', metavar='int', help='the job number', required=True)
    parser.add_argument('--dataset_name', metavar='str', help='Dataset name', required=True)
    parser.add_argument('--sample_rate', metavar='int', help='sampling rate', required=True)
    parser.add_argument('--model_name', metavar='str', help='ASR model name, e.g., whisper', required=True)
    parser.add_argument('--model_size', metavar='str', help='additional information for a model size', required=True)

    args = parser.parse_args()
    total_jobs    = int(args.total_jobs)
    job_number    = int(args.job_number)
    dataset_name  = args.dataset_name  
    sample_rate   = int(args.sample_rate)
    model_name    = args.model_name
    model_size    = args.model_size  
    print(args)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stm_ids, utterance_dict, transcript_dict = build_segment_dict(dataset_name)
    stm_ids.sort()
    start_line, end_line = split_segment_for_parallel(stm_ids, total_jobs, job_number)

    hypothesis_path = f"/share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/hypotheses/{dataset_name}"
    Path(hypothesis_path).mkdir(parents=True, exist_ok=True)
    hypothesis_file_name = hypothesis_path + f"/data.{model_name}_{model_size}_{total_jobs}_{job_number}.stm"

    save_hypotheses(model_name, model_size, dataset_name, sample_rate, stm_ids, utterance_dict, transcript_dict, start_line, end_line, hypothesis_file_name, job_number, total_jobs)
