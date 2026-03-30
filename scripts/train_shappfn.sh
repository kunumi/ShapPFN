python src/train.py \
    --model shappfn --use_shap_loss True \
    --wandb_name "shappfn" \
    --max_steps 8000 \
    --num_background_samples 4 \
    --prior_dir data/ \
    --out-dir "checkpoints/shappfn"