import torch
import torch.nn as nn
import math


def viashap_loss(
    model: nn.Module,
    x_full: torch.Tensor,         # (B, R, F)
    y_train: torch.Tensor,        # (B, R_train) or (B, R_train, 1)
    train_test_split_index: int,
    num_subsets: int = 4,         # S
    num_bg_samples: int = 4,      # K
    eps_kernel: float = 1e-8,
):
    """
    row-sampled background + K-sample MC average of logits.

    For each subset (coalition) and each test row:
      - masked features are replaced by values from a randomly sampled TRAIN row
      - do K such background draws, average logits over K
    Regularizer enforces:
      logits_avg(x^s) ≈ base(x) + sum_{kept features} phi_j(x)

    Assumes model((x_full, y_train), ...) returns:
      logits: (B, N_test, Cmax)
      base:   (B, N_test, Cmax)
      phi:    (B, N_test, F, Cmax)
    """
    device = x_full.device
    dtype = x_full.dtype

    B, R, F = x_full.shape
    R_train = train_test_split_index
    N_test = R - R_train
    S = num_subsets
    K = num_bg_samples

    # Unmasked pass (base/phi for additive approximation)
    logits_full, base_full, phi_full = model((x_full, y_train), train_test_split_index=R_train)

    if F <= 1:
        # degenerate: no meaningful subsets; return 0
        return torch.zeros((), device=device, dtype=dtype)

    # sample k in {1, ..., F-1} 
    k = torch.randint(low=1, high=F, size=(S,), device=device)  # (S,)

    ranks = torch.rand(S, F, device=device)
    idx = ranks.argsort(dim=1, descending=True)  # (S, F)
    kept = torch.zeros(S, F, device=device, dtype=torch.bool)
    pos = torch.arange(F, device=device).unsqueeze(0).expand(S, F)
    topk_pos = pos < k.unsqueeze(1)
    kept.scatter_(1, idx, topk_pos)                                   # True for kept features

    s = (~kept).to(dtype)  # (S, F)

    # Shapley kernel weights w(k) ∝ (F-1) / (C(F,k) * k * (F-k))
    kf = k.to(torch.float32)
    logC = (
        torch.lgamma(torch.tensor(float(F), device=device) + 1.0)
        - torch.lgamma(kf + 1.0)
        - torch.lgamma(torch.tensor(float(F), device=device) - kf + 1.0)
    )
    logw = math.log(max(F - 1, 1)) - logC - torch.log(kf + eps_kernel) - torch.log((F - kf) + eps_kernel)
    w = torch.exp(logw).to(dtype)  # (S,)
    w = w / (w.mean() + eps_kernel)

    x_train_rows = x_full[:, :R_train, :]
    bg_idx = torch.randint(0, R_train, size=(B, S, N_test, K), device=device)

    # gather bg rows -> (B, S, N_test, K, F)
    xtr = x_train_rows.unsqueeze(1).expand(B, S, R_train, F)
    xtrK = xtr.unsqueeze(3).expand(B, S, R_train, K, F)
    gather_idx = bg_idx.unsqueeze(-1).expand(B, S, N_test, K, F)
    bg_vals = xtrK.gather(dim=2, index=gather_idx)

    s_mask = s.unsqueeze(0).unsqueeze(2).unsqueeze(3) # (1,S,1,1,F)

    x_test = x_full[:, R_train:, :].unsqueeze(1).expand(B, S, N_test, F)     # (B,S,N_test,F)
    x_testK = x_test.unsqueeze(3).expand(B, S, N_test, K, F)                 # (B,S,N_test,K,F)

    x_masked_testK = x_testK * (1.0 - s_mask) + bg_vals * s_mask             # (B,S,N_test,K,F)

    x_train_part = x_full[:, :R_train, :].unsqueeze(1).unsqueeze(3).expand(B, S, R_train, K, F)

    x_masked_fullK = torch.cat([x_train_part, x_masked_testK], dim=2)

    x_in = x_masked_fullK.permute(0, 1, 3, 2, 4).reshape(B * S * K, R, F)

    if y_train.ndim == 2:
        y_in = y_train.unsqueeze(1).unsqueeze(2).expand(B, S, K, R_train).reshape(B * S * K, R_train)
    else:
        y_in = y_train.unsqueeze(1).unsqueeze(2).expand(B, S, K, R_train, y_train.shape[-1]).reshape(
            B * S * K, R_train, y_train.shape[-1]
        )

    logits_masked, _, _ = model((x_in, y_in), train_test_split_index=R_train)
    # logits_masked: (B*S*K, N_test, Cmax) -> (B,S,K,N_test,Cmax)
    logits_masked = logits_masked.view(B, S, K, N_test, logits_masked.shape[-1])
    logits_avg = logits_masked.mean(dim=2)

    # kept: (S,F) -> (1,S,1,F,1)
    kept_mask = kept.to(dtype).unsqueeze(0).unsqueeze(2).unsqueeze(-1)       # (1,S,1,F,1)
    phi_sum = (phi_full.unsqueeze(1) * kept_mask).sum(dim=3)                 # (B,S,N_test,Cmax)
    approx = base_full.unsqueeze(1) + phi_sum                                # (B,S,N_test,Cmax)

    err_per_subset = (approx - logits_avg).pow(2).mean(dim=(0, 2, 3))         # (S,)
    shap_reg = (err_per_subset * w).mean()

    return shap_reg
