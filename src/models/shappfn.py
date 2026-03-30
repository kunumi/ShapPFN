import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm
import shap
import copy

class ShapPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))

        self.shap_decoder = ShapDecoder(embedding_size, mlp_hidden_size, num_outputs)
        self.base_decoder = BaseDecoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)

        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)

        train_tgt = src[:, :train_test_split_index, -1, :]  # (B, N_train, E)
        task_repr = train_tgt.mean(dim=1)                   # (B, E)
        base = self.base_decoder(task_repr)
        base = base.unsqueeze(1).expand(-1, src.shape[1]-train_test_split_index, -1)     

        feature_embs = src[:, train_test_split_index:, :-1, :]  # (B, N_test, F, E)
        phi = self.shap_decoder(feature_embs)  # (B, N_test, F, C)
        logits = phi.sum(dim=2) + base  # (B, N_test, C)

        return logits, base, phi


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the linear layer that we will use to embed our features. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True, unbiased=False) + 1e-20
        x = (x-mean)/std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)

class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the linear layer that we will use to embed our targets. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Padds up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        # nan padding & nan handler instead?
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows-y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)

class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
            train_test_split_index: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0]+src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attention_between_datapoints(src[:,:train_test_split_index], src[:,:train_test_split_index], src[:,:train_test_split_index])[0]
        # test data attends to the training data
        src_right = self.self_attention_between_datapoints(src[:,train_test_split_index:], src[:,:train_test_split_index], src[:,:train_test_split_index])[0]
        src = torch.cat([src_left, src_right], dim=1)+src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src
    
class ShapDecoder(nn.Module):
    """
    Produces per-feature per-class contributions (phi).
    Input:  (B, N_test, F, E)
    Output: (B, N_test, F, Cmax)
    """
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, feat_emb: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(feat_emb)))


class BaseDecoder(nn.Module):
    """
    Produces the base term per-row (like bias), conditioned on target token embedding.
    Input:  (B, N_test, E)
    Output: (B, N_test, Cmax)
    """
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, tgt_emb: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(tgt_emb)))

class ShapPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model: ShapPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.feature_names = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        feature_names: list[str] = None,
    ):
        """stores X_train/y_train, computes num_classes; optionally stores feature names"""
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

        if feature_names is not None:
            self.feature_names = list(feature_names)

        if self.feature_names is not None:
            n_features = X_train.shape[1]
            if len(self.feature_names) != n_features:
                raise ValueError(
                    f"feature_names length ({len(self.feature_names)}) must match "
                    f"number of features in X_train ({n_features})."
                )


        return self

    def explain(self, X_test: np.array) -> shap.Explanation:
        """
        Returns a single shap.Explanation object:
          - values:      (n_test, n_features, n_classes)  (logit contributions)
          - base_values: (n_test, n_classes)              (logit base term)
          - data:        (n_test, n_features)
        And satisfies: logits = base_values + values.sum(axis=1)
        """
        x = np.concatenate((self.X_train, X_test), axis=0)
        y = self.y_train

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # (1, n_all, F)
            y_t = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)  # (1, n_train)
            logits, base, phi = self.model((x_t, y_t), train_test_split_index=len(self.X_train))

            # slice out test part and drop batch dim
            phi = phi.squeeze(0)            # (n_test, F, Cmax)
            base = base.squeeze(0)          # (n_test, Cmax)
            logits = logits.squeeze(0)      # (n_test, Cmax)

            # cut to actually-observed classes
            phi = phi[:, :, :self.num_classes]       # (n_test, F, C)
            base = base[:, :self.num_classes]        # (n_test, C)
            logits = logits[:, :self.num_classes]    # (n_test, C)

            phi_np = phi.detach().cpu().numpy()
            base_np = base.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()

        # Build SHAP Explanation
        exp = shap.Explanation(
            values=phi_np,                 # (n_test, n_features, n_classes)
            base_values=base_np,           # (n_test, n_classes)
            data=X_test,                   # (n_test, n_features)
            feature_names=self.feature_names
        )
        exp.output_values = logits_np      # (n_test, n_classes)

        return exp

    def predict_logits(self, X_test: np.array) -> np.array:
        """
        Returns raw, pre-softmax logits in logit space.
        Shape: (n_test, num_classes)
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            logits, _, _ = self.model((x, y), train_test_split_index=len(self.X_train))
            logits = logits.squeeze(0)
            logits = logits[:, :self.num_classes]
            return logits.to("cpu").numpy()
        
    def predict_proba(self, X_test: np.array) -> np.array:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out, base, phi = self.model((x, y), train_test_split_index=len(self.X_train))  # remove batch size 1
            out = out.squeeze(0)
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.array) -> np.array:
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)



class ShapPFNMultiClass:
    """
    One-vs-all (OvA) multiclass wrapper around a *binary* ShapPFN model.

    Assumptions:
      - `binary_model` is a ShapPFNModel with num_outputs == 2 (neg/pos logits).
      - Under the hood, each class k gets its own binary problem: y_bin = 1{y == k}.
      - We convert each binary model's 2-logit output into a single OvA "score logit":
            score_k = logit_pos - logit_neg
        and then turn scores into multiclass probabilities via softmax across classes.

    Provides:
      - predict_logits: (n_test, K) OvA scores
      - predict_proba:  (n_test, K) softmax(scores)
      - explain:        shap.Explanation with:
            values:      (n_test, n_features, K)  contributions in score-logit space
            base_values: (n_test, K)
            data:        X_test
        satisfying: scores = base_values + values.sum(axis=1)
    """
    def __init__(self, binary_model, device: torch.device):
        self.binary_model_template = binary_model
        self.device = device
        self.binary_clfs = None
        self.num_classes = None

    def _clone_binary_clf(self):
        m = copy.deepcopy(self.binary_model_template).to(self.device)
        return ShapPFNClassifier(m, self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train.astype(int)

        self.num_classes = int(np.max(self.y_train)) + 1

        self.binary_clfs = [self._clone_binary_clf() for _ in range(self.num_classes)]

        # fit each binary classifier with y_bin = (y==k)
        for k, clf in enumerate(self.binary_clfs):
            y_bin = (self.y_train == k).astype(int)
            clf.fit(self.X_train, y_bin)

        return self

    @staticmethod
    def _to_score_from_2logits(logits_2: np.ndarray) -> np.ndarray:
        """
        logits_2: (n_test, 2) -> score: (n_test,)
        score = logit_pos - logit_neg
        """
        return logits_2[:, 1] - logits_2[:, 0]

    @staticmethod
    def _to_score_phi_base(exp_bin: shap.Explanation):
        """
        exp_bin.values:      (n_test, F, 2)
        exp_bin.base_values: (n_test, 2)
        -> score space:
           phi_score:  (n_test, F)
           base_score: (n_test,)
        where score = base_score + phi_score.sum(axis=1)
        """
        phi = exp_bin.values           # (n, F, 2)
        base = exp_bin.base_values     # (n, 2)
        phi_score = phi[:, :, 1] - phi[:, :, 0]
        base_score = base[:, 1] - base[:, 0]
        return phi_score, base_score

    def predict_logits(self, X_test: np.ndarray) -> np.ndarray:
        """
        Returns OvA scores in logit space. Shape: (n_test, K)
        """
        if self.binary_clfs is None:
            raise RuntimeError("Call fit() before predict_logits().")

        scores = []
        for clf in self.binary_clfs:
            logits_2 = clf.predict_logits(X_test)  # (n_test, 2)
            scores.append(self._to_score_from_2logits(logits_2))

        return np.stack(scores, axis=1)  # (n_test, K)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Multiclass probabilities from softmax over OvA scores. Shape: (n_test, K)
        """
        scores = self.predict_logits(X_test)
        
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test).argmax(axis=1)

    def explain(self, X_test: np.ndarray) -> shap.Explanation:
        """
        Returns a single multiclass SHAP Explanation in OvA score-logit space:
          - values:      (n_test, n_features, K)
          - base_values: (n_test, K)
          - data:        (n_test, n_features)

        And satisfies: scores = base_values + values.sum(axis=1)
        where scores are OvA scores (logit_pos - logit_neg) per class.
        """
        if self.binary_clfs is None:
            raise RuntimeError("Call fit() before explain().")

        phi_all = []
        base_all = []
        score_all = []

        for clf in self.binary_clfs:
            exp_bin = clf.explain(X_test)  # values (n,F,2), base (n,2)

            phi_score, base_score = self._to_score_phi_base(exp_bin)
            phi_all.append(phi_score)      # (n,F)
            base_all.append(base_score)    # (n,)

            # also store the model outputs in score space
            # exp_bin.output_values was set to (n,2) logits in ShapPFNClassifier.explain
            if hasattr(exp_bin, "output_values") and exp_bin.output_values is not None:
                logits_2 = exp_bin.output_values  # (n,2)
                score_all.append(self._to_score_from_2logits(logits_2))
            else:
                score_all.append(base_score + phi_score.sum(axis=1))

        # stack into multiclass tensors
        # phi_mc: (n, F, K), base_mc: (n, K), scores_mc: (n, K)
        phi_mc = np.stack(phi_all, axis=2)
        base_mc = np.stack(base_all, axis=1)
        scores_mc = np.stack(score_all, axis=1)

        exp = shap.Explanation(
            values=phi_mc,
            base_values=base_mc,
            data=X_test,
        )
        exp.output_values = scores_mc  # (n, K) OvA scores

        return exp

