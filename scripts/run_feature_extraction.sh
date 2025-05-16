#!/usr/bin/bash

TOTAL_JOBS=10

for FEATURE_NAME in hubert # xlmr
do
for DATASET_NAME in tl3_dev # tl3_test tl3_train 
do
for JOB_NUMBER in {1..10}
do
SH_NM=feature_extraction_${FEATURE_NAME}_${DATASET_NAME}_${TOTAL_JOBS}_${JOB_NUMBER}
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
python -u \${PWD}/feature_extraction.py \
--total_jobs    10 \
--job_number    1 \
--dataset_name  ${DATASET_NAME} \
--feature_name  ${FEATURE_NAME} \
--training_type ft \
--sample_rate   16000 \
--feature_layer 24 \
--feature_level sequence \
--feature_size  large
EOF

GPU_TYPE=3060
JID=`/share/spandh.ami1/sw/mini/jet/latest/tools/submitjob  \
     -g1 -M2 -q NORMAL \
     -o -l gputype=${GPU_TYPE} -eo \
     ${LOG_DIR}/${SH_NM}.log \
     ${SCRPT_DIR}/${SH_NM}.sh | tail -1`
echo "scancel ${JID}"
done
done
done
