import time
import torch
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics.pairwise import cosine_similarity

from models import ShapPFNClassifier, ShapPFNMultiClass, ShapPFNModel
from eval.eval_utils import get_feature_preprocessor, get_openml_datasets


torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# Helpers
# ----------------------------
def cosine_similarity_shap(a, b):
    similarities = [
        cosine_similarity(
            a[i].reshape(1, -1),
            b[i].reshape(1, -1)
        )[0][0]
        for i in range(a.shape[0])
    ]
    return np.mean(similarities)

def spearman_shap(a, b):
    spearman_corrs = [
        spearmanr(a[i], b[i])[0]
        for i in range(a.shape[0])
    ]
    return np.mean(spearman_corrs)

def r2_shap(a, b):
    r2_values = [
        r2_score(a[i], b[i])
        for i in range(a.shape[0])
    ]
    return np.mean(r2_values)


def main(out_dir, file_name, model_path, num_features):
    model = ShapPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()


    results = []
    DATASETS = get_openml_datasets(max_features_eval=num_features, new_instances_eval=200, target_classes_filter=10)
    for dataset_name, (X, y) in DATASETS.items():

        clf = ShapPFNMultiClass(model, device=torch.device("cpu"))
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        preprocessor = get_feature_preprocessor(X)
        X = preprocessor.fit_transform(X)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        except Exception as e:
            print(f"Error splitting dataset {dataset_name}: {e}")
            continue

        clf.fit(X_train, y_train)

        # Internal SHAP -> (n, d, C)
        start_model = time.perf_counter()
        shap_internal_full = clf.explain(X_test)
        end_model = time.perf_counter()
        time_model = end_model - start_model
        shap_internal_full = np.asarray(shap_internal_full.values)

        # KernelExplainer SHAP -> Explanation
        explainer = shap.KernelExplainer(clf.predict_logits, X_train[:100])
        start_kernel = time.perf_counter()
        shap_kernel_exp = explainer(X_test)
        end_kernel = time.perf_counter()
        time_kernel = end_kernel - start_kernel

        # Explanation.values -> (n, d, C)
        shap_kernel_full = np.asarray(shap_kernel_exp.values)

        n, d, C = shap_internal_full.shape

        shap_internal = shap_internal_full.reshape(n, d*C) # Compare all the classes
        shap_kernel = shap_kernel_full.reshape(n, d*C)

        if shap_internal.shape != shap_kernel.shape:
            raise ValueError(
                f"Shape mismatch: internal {shap_internal.shape} vs kernel {shap_kernel.shape}"
            )

        n, d = shap_internal.shape
        print(f"Compared SHAP arrays with shape: (n={n}, d={d})")

        # ----------------------------
        # Metrics
        # ----------------------------
        global_cos = cosine_similarity_shap(shap_internal, shap_kernel)
        global_spear = spearman_shap(shap_internal, shap_kernel)
        global_r2 = r2_shap(shap_internal, shap_kernel)


        print(f"\nDataset: {dataset_name}")

        print("\n=== R2 score ===")
        print(f"Global (flattened): {global_r2:.6f}")
        print("\n=== Cosine similarity ===")
        print(f"Global (flattened): {global_cos:.6f}")
        print("\n=== Spearman correlation ===")
        print(f"Global (flattened): {global_spear:.6f}")

        results.append({
            "dataset": dataset_name,
            "n": int(n),
            "d": int(d),
            "r2_global": global_r2,
            "cos_global": global_cos,
            "spearman_global": global_spear,
            "time_model": time_model,
            "time_kernel": time_kernel
        })

    results_df = pd.DataFrame(results)

    mean_row = {
        "dataset": "MEAN",
        "n": results_df["n"].mean(),
        "d": results_df["d"].mean(),
        "r2_global": results_df["r2_global"].mean(),
        "cos_global": results_df["cos_global"].mean(),
        "spearman_global": results_df["spearman_global"].mean(),
        "time_model": results_df["time_model"].mean(),
        "time_kernel": results_df["time_kernel"].mean(),
    }

    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row])],
        ignore_index=True
    )

    results_df.to_csv(f"{out_dir}/{file_name}", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to score the shap values generated by the model")
    parser.add_argument("--shappfn-path", type=str, help="The shappfn model path")
    parser.add_argument("--out-dir", type=str, help="The output dir")
    parser.add_argument("--file_name", type=str, default="shap_values_comparison.csv", help="The output file name")
    parser.add_argument("--num_features", type=int, default=10, help="Number of features to use")
    
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main(out_dir, args.file_name, args.shappfn_path, args.num_features)
