
#!/bin/bash

# Datasets and their CUDA device
datasets=(
    "Games_5core 0"
    "Arts_5core 0"
    "Movies_5core 0"
    # "Goodreads 1"
    # "Sports_5core 1"
    # "Baby_5core 1"
)

# Recommender fixed parameters
run_id="Eval_Embeddings"
model="SASRec"
# model="GRU4Rec"
dr=0.3
lr=1.0e-3
wd=1.0e-4
loss_type="ce"

# Extract embeddings if missing
extract_if_missing() {
    local dataset=$1
    local cuda_device=$2
    local model_name=$3
    local save_info=$4

    local emb_file="./item_info/${dataset}/${save_info}_title_item_embs.npy"

    if [ ! -f "$emb_file" ]; then
        echo "Embedding file $emb_file not found. Extracting..."
        CUDA_VISIBLE_DEVICES=$cuda_device python extract_embeddings.py \
            --model_name="$model_name" \
            --dataset="$dataset" \
            --batch_size=64 \
            --prompt_type=title \
            --save_info="$save_info"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to extract embedding for $model_name on $dataset"
            return 1
        fi
        echo "✅ Extraction completed: $emb_file"
    fi
    return 0
}

# Run experiments for all datasets simultaneously
for dataset_entry in "${datasets[@]}"; do
    (
        IFS=' ' read -r dataset cuda_device <<< "$dataset_entry"

        # Models to run: "encoder_name save_prefix"
        models_to_run=(
            # "Blair Blair"
            # "BGE BGE"
            # "BERT BERT"
            "GTE_7B GTE_7B"
            # "EasyRec EasyRec"
            # "LLM2Vec LLM2Vec"
        )

        for model_entry in "${models_to_run[@]}"; do
            IFS=' ' read -r model_name save_info <<< "$model_entry"

            extract_if_missing "$dataset" "$cuda_device" "$model_name" "$save_info" || continue

            embs="./item_info/${dataset}/${save_info}_title_item_embs.npy"
            echo "Running evaluation with dataset=$dataset, device=$cuda_device, embeddings=$embs"
            CUDA_VISIBLE_DEVICES=$cuda_device python evaluate.py \
                --model="$model" \
                --dataset="$dataset" \
                --lr="$lr" \
                --weight_decay="$wd" \
                --embedding="$embs" \
                --dropout="$dr" \
                --loss_type="$loss_type" \
                --run_id="$run_id" \
                --num_epochs=150
        done
    ) &
done

wait
