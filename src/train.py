import math
import random
import time

import numpy as np
import schedulefree
import torch
from torch import nn
from torch.utils.data import DataLoader
import functools
from pathlib import Path
from tabicl.prior.genload import PriorDataset, LoadPriorDataset


from models import ShapPFNModel, ShapPFNClassifier, NanoTabPFNModel, NanoTabPFNClassifier
from eval.eval_utils import eval_model, get_openml_datasets
from utils import build_parser
from viashap import viashap_loss
try:
    import wandb
except ImportError:
    wandb = None

seed = 0

def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(seed)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device


def train(model: ShapPFNModel, prior: DataLoader,
          lr: float = 1e-4, use_shap_loss: bool = True, 
          shap_loss_weight: float = 2.0, warmup_steps=1000,
          num_subsets: int = 4, num_background_samples: int = 4,
          device: torch.device = None, steps_per_eval=10,
          eval_func=None, max_steps=None, wandb_run=None):
    
    if not device:
        device = get_default_device()
    if wandb_run is None and wandb is not None and wandb.run is not None:
        wandb_run = wandb.run

    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history=[]
    prior_iter = iter(prior)
    try:
        for step in range(max_steps if max_steps is not None else 100000):
            step_start_time = time.time()
            full_data = next(prior_iter)
            x_full, y_full, d, seq_lens, train_sizes = full_data
            
            if len(torch.unique(seq_lens)) > 1:
                raise ValueError("All datasets in the batch must have the same sequence length.")

            if len(torch.unique(train_sizes)) > 1:
                raise ValueError("All datasets in the batch must have the same training size.")
            
            if not (torch.isfinite(x_full).all() and torch.isfinite(y_full).all()):
                continue
            
            train_test_split_index = train_sizes[0].item()

            data = (x_full.to(device),
                    y_full[:, :train_test_split_index].to(device))
            targets = y_full.to(device)

            output, base, phi = model(data, train_test_split_index=train_test_split_index)
            targets = targets[:, train_test_split_index:]

            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            ce_loss = criterion(output, targets).mean()
            shap_loss = 0
            if use_shap_loss:
                shap_loss = viashap_loss(
                    model,
                    x_full=data[0],
                    y_train=data[1],
                    train_test_split_index=train_test_split_index,
                    num_subsets=num_subsets,
                    num_bg_samples=num_background_samples,
                )
                if warmup_steps > 0:
                    corrected_weight = shap_loss_weight * min(1, step / warmup_steps) 
                else:
                    corrected_weight = shap_loss_weight
                loss = ce_loss + corrected_weight * shap_loss
            else:
                loss = ce_loss
            loss.backward()
            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": total_loss,
                        "train/ce_loss": ce_loss.detach().cpu().item(),
                        "train/shap_loss": shap_loss.detach().cpu().item() if use_shap_loss else 0.0,
                        "train/shap_loss_weight": corrected_weight if use_shap_loss else 0.0,
                        "train/step_time": step_train_duration,
                        "train/time": train_time,
                    },
                    step=step,
                )

            # evaluate
            if (step % steps_per_eval == steps_per_eval-1 and eval_func is not None) or (max_steps is not None and step >= (max_steps-1)):
                model.eval()
                optimizer.eval()

                if isinstance(model, NanoTabPFNModel):
                    classifier = NanoTabPFNClassifier(model, device)
                else:
                    classifier = ShapPFNClassifier(model, device)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                if wandb_run is not None:
                    wandb_run.log({f"eval/{k}": v for k, v in scores.items()}, step=step)
                score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")

                model.train()
                optimizer.train()
            elif step % steps_per_eval == steps_per_eval-1 and eval_func is None:
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")
    except KeyboardInterrupt:
        pass

    return model, eval_history

if __name__ == "__main__":

    parser = build_parser()
    config = parser.parse_args()

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_default_device()

    if config.model == "nano_tab_pfn":
        model = NanoTabPFNModel(
            embedding_size=config.embedding_size,
            num_attention_heads=config.num_attention_heads,
            mlp_hidden_size=config.mlp_hidden_size,
            num_layers=config.num_layers,
            num_outputs=config.num_outputs,
        )
    else:
        model = ShapPFNModel(
            embedding_size=config.embedding_size,
            num_attention_heads=config.num_attention_heads,
            mlp_hidden_size=config.mlp_hidden_size,
            num_layers=config.num_layers,
            num_outputs=config.num_outputs,
        )

    eval_model_partial = None
    if config.eval_openml:
        datasets = get_openml_datasets(
            max_features_eval=15,
            new_instances_eval=1024,
            target_classes_filter=2,
        )
        eval_model_partial = functools.partial(eval_model, datasets=datasets)

    if config.prior_dir is not None:
        print("Using pre-generated data")
        dataset = LoadPriorDataset(
            data_dir=config.prior_dir,
            batch_size=config.prior_batch_size,
            device=config.prior_device,
            )
    else:
        dataset = PriorDataset(
            batch_size=config.prior_batch_size,
            batch_size_per_gp=config.batch_size_per_gp,
            min_features=config.min_features,
            max_features=config.max_features,
            max_classes=config.max_classes,
            max_seq_len=config.max_seq_len,
            min_train_size=config.min_train_size,
            max_train_size=config.max_train_size,
            prior_type=config.prior_type,
            device=config.prior_device,
            n_jobs=config.n_jobs,
        )

    dataloader_kwargs = dict(
        batch_size=config.loader_batch_size,
        shuffle=config.shuffle,
        pin_memory=config.pin_memory,
        pin_memory_device=config.pin_memory_device,
        num_workers=config.num_workers,
    )
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.persistent_workers
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor
    else:
        dataloader_kwargs["persistent_workers"] = False

    prior = DataLoader(dataset, **dataloader_kwargs)

    wandb_run = None
    if config.wandb and wandb is not None:
        wandb_name = config.wandb_name or config.model
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=wandb_name,
            config={
                "seed": seed,
                "device": device,
                "model": config.model,
                "embedding_size": config.embedding_size,
                "num_attention_heads": config.num_attention_heads,
                "mlp_hidden_size": config.mlp_hidden_size,
                "num_layers": config.num_layers,
                "num_outputs": config.num_outputs,
                "prior_batch_size": config.prior_batch_size,
                "batch_size_per_gp": config.batch_size_per_gp,
                "min_features": config.min_features,
                "max_features": config.max_features,
                "max_classes": config.max_classes,
                "max_seq_len": config.max_seq_len,
                "min_train_size": config.min_train_size,
                "max_train_size": config.max_train_size,
                "prior_type": config.prior_type,
                "prior_device": config.prior_device,
                "n_jobs": config.n_jobs,
                "loader_batch_size": config.loader_batch_size,
                "shuffle": config.shuffle,
                "pin_memory": config.pin_memory,
                "pin_memory_device": config.pin_memory_device,
                "num_workers": config.num_workers,
                "persistent_workers": config.persistent_workers,
                "prefetch_factor": config.prefetch_factor,
                "lr": config.lr,
                "steps_per_eval": config.steps_per_eval,
                "use_shap_loss": config.use_shap_loss,
                "shap_loss_weight": config.shap_loss_weight,
                "warmup_steps": config.warmup_steps,
                "num_subsets": config.num_subsets,
                "num_background_samples": config.num_background_samples,
                "max_steps": config.max_steps,
                "eval_openml": config.eval_openml,
            },
        )
    model, history = train(
        model,
        prior,
        lr=config.lr,
        steps_per_eval=config.steps_per_eval,
        use_shap_loss=config.use_shap_loss,

        shap_loss_weight=config.shap_loss_weight,
        num_subsets=config.num_subsets,
        num_background_samples=config.num_background_samples,
        warmup_steps=config.warmup_steps,

        device=device,
        eval_func=eval_model_partial,
        wandb_run=wandb_run,
        max_steps=config.max_steps,
    )

    name = wandb_run.name if wandb_run is not None else time.strftime("%Y%m%d_%H%M%S")
    if wandb_run is not None:
        wandb_run.finish()

    torch.save(model.state_dict(), f"{out_dir}/{config.model}_{name}.pth")
    
