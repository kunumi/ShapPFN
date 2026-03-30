import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        raise argparse.ArgumentTypeError("Expected a boolean value.")
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}. Use true/false.")


def optional_int(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid int value: {value!r}.") from exc


def build_parser():
    """Build parser with all TabICL training arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection / architecture.
    parser.add_argument(
        "--model",
        default="shappfn",
        type=str,
        choices=["shappfn", "nano_tab_pfn"],
        help="The model architecture to train",
    )
    parser.add_argument("--embedding_size", default=96, type=int)
    parser.add_argument("--num_attention_heads", default=4, type=int)
    parser.add_argument("--mlp_hidden_size", default=192, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--num_outputs", default=2, type=int)

    # PriorDataset / synthetic task sampling.
    parser.add_argument("--prior_batch_size", default=32, type=int)
    parser.add_argument("--batch_size_per_gp", default=4, type=int)
    parser.add_argument("--min_features", default=2, type=int)
    parser.add_argument("--max_features", default=5, type=int)
    parser.add_argument("--max_classes", default=2, type=int)
    parser.add_argument("--max_seq_len", default=200, type=int)
    parser.add_argument("--min_train_size", default=0.1, type=float)
    parser.add_argument("--max_train_size", default=0.9, type=float)
    parser.add_argument("--prior_type", default="mix_scm", type=str)
    parser.add_argument("--prior_device", default="cpu", type=str)
    parser.add_argument("--n_jobs", default=-1, type=int)

    # PyTorch DataLoader for the prior.
    parser.add_argument("--prior_dir", default=None, type=str)
    parser.add_argument("--loader_batch_size", default=None, type=optional_int)
    parser.add_argument("--shuffle", default=False, type=str2bool)
    parser.add_argument("--pin_memory", default=True, type=str2bool)
    parser.add_argument("--pin_memory_device", default="cuda", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--persistent_workers", default=True, type=str2bool)
    parser.add_argument("--prefetch_factor", default=2, type=int)

    # Training loop.
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--steps_per_eval", default=80, type=int)
    parser.add_argument("--use_shap_loss", default=False, type=str2bool)

    parser.add_argument("--shap_loss_weight", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=1200, type=int)
    parser.add_argument("--num_subsets", default=4, type=int)
    parser.add_argument("--num_background_samples", default=8, type=int)

    parser.add_argument("--max_steps", default=None, type=int)

    parser.add_argument("--eval_openml", default=True, type=str2bool)
    parser.add_argument("--wandb", default=True, type=str2bool)
    parser.add_argument("--wandb_project", default="shappfn", type=str)
    parser.add_argument("--wandb_name", default='train', type=str)

    parser.add_argument("--out-dir", default=None, type=str)

    return parser
