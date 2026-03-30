from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
import openml
from openml.tasks import TaskType
import numpy as np
import pandas as pd
import torch 
import time
from models import ShapPFNModel, ShapPFNClassifier


_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def eval_model(model, datasets):
    """Evaluates a model on multiple datasets and returns metrics"""
    
    start_time = time.perf_counter()
    
    metrics = {}
    for dataset_name, (X, y) in datasets.items():
        targets = []
        probabilities = []
        
        for train_idx, test_idx in _skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            targets.append(y_test)
            
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)
            
            if y_proba.shape[1] == 2:  # binary classification
                y_proba = y_proba[:, 1]
            
            probabilities.append(y_proba)
    
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)
        
        metrics[f"{dataset_name}/ROC AUC"] = roc_auc_score(
            targets, probabilities, multi_class="ovr"
        )
    
    end_time = time.perf_counter()
    
    # Average metrics across datasets
    metric_names = list({key.split("/")[-1] for key in metrics.keys()})
    for metric_name in metric_names:
        avg_metric = np.mean(
            [metrics[key] for key in metrics.keys() if key.endswith(metric_name)]
        )
        metrics[metric_name] = float(avg_metric)
    
    metrics["evaluation_time_seconds"] = end_time - start_time
    
    return metrics



def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(
            lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x,
            feature_names_out="one-to-one"
        )),
        ("to_numeric", FunctionTransformer(
            lambda x: x.apply(pd.to_numeric, errors="coerce").to_numpy(),
            feature_names_out="one-to-one"
        )),
    ])
    
    cat_transformer = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor

def get_openml_datasets(
        max_features_eval: int = 10, 
        new_instances_eval: int = 200, 
        target_classes_filter: int = 2,
        **kwargs,
        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load OpenML tabarena datasets with at most `max_features` features and subsampled (stratified) to `new_instances` instances.
    """
    suite_id = 99
    suite = openml.study.get_suite(suite_id)

    task_ids = suite.tasks
    datasets = {}
    used_ids = []
    metadata_records = []


    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue  # skip task, only classification
        dataset = task.get_dataset(download_data=False)

        if dataset.qualities["NumberOfFeatures"] > max_features_eval or (dataset.qualities["NumberOfClasses"] > target_classes_filter) or dataset.qualities["PercentageOfInstancesWithMissingValues"] > 0 or dataset.qualities["MinorityClassPercentage"] < 2.5:
            continue
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        if new_instances_eval < len(y):
            _, X_sub, _, y_sub = train_test_split(
                X, y,
                test_size=new_instances_eval,
                stratify=y,
                random_state=0,
            )
        else:
            X_sub = X
            y_sub = y
        
        X = X_sub.to_numpy(copy=True)
        y = y_sub.to_numpy(copy=True)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        preprocessor = get_feature_preprocessor(X)
        X = preprocessor.fit_transform(X)
        datasets[dataset.name] = (X, y)
        used_ids.append(task_id)

        metadata_records.append({
            "dataset_name": dataset.name,
            "task_id": task_id,
            "n_features": dataset.qualities["NumberOfFeatures"],
            "n_classes": len(np.unique(y)),
        })

        

        print(f"Added datased: {dataset.name}, with id {task_id}")
    metadata_df = pd.DataFrame(metadata_records)
    print(f"Using datasets with ids {used_ids} for eval")
    return datasets

