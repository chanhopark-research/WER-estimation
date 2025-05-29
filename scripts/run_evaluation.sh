#!/usr/bin/bash

EXPERIMENT_NAME=Fe-WER

for LAYER1_SIZE in 600
do
for LAYER2_SIZE in 32
do
for LEARNING_RATE in 0.0007
do
SH_NM=evaluation_${EXPERIMENT_NAME}_${LAYER1_SIZE}_${LAYER2_SIZE}_${LEARNING_RATE}
SCRPT_DIR="${PWD}/${SH_NM}"
LOG_DIR="${SCRPT_DIR}"
mkdir -p ${SCRPT_DIR}

cat << EOF > ${SCRPT_DIR}/${SH_NM}.sh
#!/usr/bin/bash

# defining conda and python dir
CONDA=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
CONDA_ENV=dt-cp-py3.9

# activate conda env
source \${CONDA}/bin/activate \${CONDA_ENV}
PYTHON=\${CONDA}/envs/\${CONDA_ENV}/bin/python

export LD_LIBRARY_PATH=\${CONDA}/envs/\${CONDA_ENV}/lib:\${LD_LIBRARY_PATH}

# for GPU
export sw=/share/mini1/sw
export cuda_version=cuda11.3
export PATH=\${sw}/std/cuda/\${cuda_version}/x86_64/bin:\${PATH}
export CUDADIR=\${sw}/std/cuda/\${cuda_version}/x86_64
export LD_LIBRARY_PATH=\${sw}/std/cuda/\${cuda_version}/x86_64/lib64:\${sw}/std/cuda/\${cuda_version}/x86_64/lib64/stubs:\${sw}/std/cuda/\${cuda_version}/x86_64/lib:\${LD_LIBRARY_PATH}
# for libcudart.so.10.2
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/share/mini1/sw/std/cuda/cuda10.2/x86_64/targets/x86_64-linux/lib

# for CUDNN
export LD_LIBRARY_PATH=/share/mini1/sw/std/cuda/cuda11.6/cudnn-linux-x86_64-8.8.0.121_cuda11-archive/lib:\${LD_LIBRARY_PATH}

export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

export KALDI_ROOT=/share/mini1/sw/mini/miniframework/latest/tools/kaldi

# -u for python to print stdout
python -u \${PWD}/evaluation.py \
--base_path /share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation \
--test_dataset_name tl3_test \
--hypothesis_name whisper_large \
--utterance_encoder_name hubert-pt-24-sequence.large \
--transcript_encoder_name xlmr-pt-24-sequence.large \
--batch_size 64 \
--num_workers 0 \
--max_duration 10 \
--model_path /share/mini1/res/t/asr/multi/multi-en/acsw/selft/opensource/WER-estimation/models/${EXPERIMENT_NAME}_${LAYER1_SIZE}_${LAYER2_SIZE}_${LEARNING_RATE} \
--layer_sizes 2048 ${LAYER1_SIZE} ${LAYER2_SIZE} 1 \
--exp_path ${SCRPT_DIR}
EOF

#GPU_TYPE=A6000
GPU_TYPE=3090
JID=`/share/spandh.ami1/sw/mini/jet/latest/tools/submitjob  \
     -g1 -M2 -q NORMAL \
     -o -l gputype=${GPU_TYPE} -eo \
     ${LOG_DIR}/${SH_NM}.log \
     ${SCRPT_DIR}/${SH_NM}.sh | tail -1`
echo "scancel ${JID}"

done
done
done
