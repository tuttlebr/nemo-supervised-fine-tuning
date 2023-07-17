#!/bin/bash
set -e
set -o pipefail

# Download weights from HuggingFace Transformers if not present.
if [[ -f /models/nemo/GPT-2B-001_bf16_tp1.nemo ]]
then 
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] /models/nemo/GPT-2B-001_bf16_tp1.nemo exists!"
else
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] Missing GPT-2B-001_bf16_tp1.nemo, downloading..."
    mkdir -p /models/nemo
    curl -L https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo \
        --output /models/nemo/GPT-2B-001_bf16_tp1.nemo
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] GPT-2B-001_bf16_tp1.nemo download completed!"
fi


# Download Dolly dataset from DataBricks on HuggingFace if not present.
if [[ -f /data/databricks-dolly-15k.jsonl ]]
then
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] /data/databricks-dolly-15k.jsonl exists."
else
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] Missing /data/databricks-dolly-15k.jsonl, downloading..."
    curl -L https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl \
        --output /data/databricks-dolly-15k.jsonl
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] /data/databricks-dolly-15k.jsonl download completed!"
fi

# Preprocess Dolly dataset if not present.
if [[ -f /data/databricks-dolly-15k-output.jsonl ]]
then
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] /data/databricks-dolly-15k-output.jsonl exists."
else
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] Missing /data/databricks-dolly-15k-output.jsonl, preprocessing..."
    python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py \
        --input /data/databricks-dolly-15k.jsonl
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] /data/databricks-dolly-15k-output.jsonl preprocessing completed!"
fi


# Split Dolly dataset if not present.
if [[ -f /data/validate/databricks-dolly-15k-validate.jsonl ]]
then
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] /data/validate/databricks-dolly-15k-validate.jsonl exists."
else
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I]  Missing /data/validate/databricks-dolly-15k-validate.jsonl, splitting..."
    mkdir -p /data/train /data/validate
    split -l 10500 /data/databricks-dolly-15k-output.jsonl --additional-suffix=".jsonl" databricks-dolly-15k-
    mv databricks-dolly-15k-aa.jsonl /data/train/databricks-dolly-15k-train.jsonl
    mv databricks-dolly-15k-ab.jsonl /data/validate/databricks-dolly-15k-validate.jsonl
    echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I]  /data/validate/databricks-dolly-15k-validate.jsonl splitting completed!"
fi


echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] Beginning Supervised Finetuning."
python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
    trainer.precision=bf16 \
    trainer.max_steps=1000 \
    trainer.devices=${TRAINER_DEVICES} \
    trainer.val_check_interval=200 \
    model.megatron_amp_O2=True \
    model.restore_from_path=/models/nemo/GPT-2B-001_bf16_tp1.nemo \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.micro_batch_size=1 \
    model.optim.lr=5e-6 \
    model.answer_only_loss=True \
    model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.train_ds.file_names=${TRAIN} \
    model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
    model.data.validation_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.validation_ds.file_names=${VALID} \
    model.data.validation_ds.names=${VALID_NAMES} \
    model.data.test_ds.file_names=${TEST} \
    model.data.test_ds.names=${TEST_NAMES} \
    model.data.test_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.train_ds.num_workers=0 \
    model.data.validation_ds.num_workers=0 \
    model.data.test_ds.num_workers=0 \
    model.data.validation_ds.metric.name=loss \
    model.data.test_ds.metric.name=loss \
    exp_manager.create_wandb_logger=False \
    exp_manager.explicit_log_dir=/results \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.monitor=validation_loss \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True

echo "[`date "+%m/%d/%Y-%H:%M:%S"`] [NEMO] [I] Completed Supervised Finetuning."