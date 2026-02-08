# training/models/rf/train_re.py
# Run: python -u -m training.models.rf.train_re
# Run: MLFLOW_EXPERIMENT_NAME="rf_re" python -u -m training.models.rf.train_re
# Docker: docker exec -it -u root priceless_chaplygin /bin/bash



import os
import mlflow
import numpy as np
import torch

from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


import mlflow.sklearn

from transformers import AutoModel, AutoTokenizer

from training.mlflow.config import setup_mlflow_from_env
from training.mlflow.logger import log_eval_results
from training.features.build_data.build_re_dataset import build_re_datasets
from training.features.vectorize.utils import build_label_maps
from training.features.vectorize.re_vectorize import build_xy_ml
from training.mlflow.registry import register_and_promote
from training.evaluation.svm_metrics import full_evaluation #### evaluation
from training.rules.post_rules import apply_post_rules



PHOBERT_NAME = os.getenv("PHOBERT_NAME", "vinai/phobert-base")



def run_train_rf(
    samples: list[dict],
    *,
    max_len: int = 256,
    seed: int = 42,
):
    print(f"🔹 Got {len(samples)} samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME, use_fast=False)
    model = AutoModel.from_pretrained(PHOBERT_NAME).to(device)
    model.eval()

    # label map
    id2label, label2id = build_label_maps(samples)
    
    print(f"🔹 Started vectorizing")
    # vectorize
    X, y, idx = build_xy_ml(
        samples,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        label2id=label2id,
    )
    print(f"🔹 Finished vectorizing")

    # ===== SPLIT 1 LẦN =====
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    print("🔍 RF hyperparameter tuning:") ###

    best_f1 = 0.0 ###
    best_clf = None ###

    # model
    clf = RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_leaf=16,
                random_state=seed,
                max_features= 'sqrt',
                n_jobs=-1,
                class_weight="balanced",
            )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # post-rule 
    print('Applying post-rules')
    y_pred_postrule = apply_post_rules(
    samples=samples,
    y_pred=y_pred,
    id2label=id2label,
    idx_test=idx_test
)
    print("\n📊 Classification report (BEST RF):")
    print(
    classification_report(
        y_test,
        y_pred_postrule,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4,
    )
)

    return best_clf, tokenizer, model, (
        X_test,
        y_test,
        y_pred_postrule,
        id2label,
        label2id,
        idx_test,
    )



# =====================================================
# Main
# =====================================================
def main():
    print("🚀 Starting train_re script")
    # 1) Load env
    load_dotenv()

    # 2) Setup MLflow
    setup_mlflow_from_env()

    # 3) Start run
    run_name = os.getenv("MLFLOW_RUN_NAME", "re_rf")
    with mlflow.start_run(run_name=run_name) as run:
        # ===== DATA =====
        re_dataset = build_re_datasets()

        # ===== TRAIN =====
        clf, tokenizer, model, (
            X_test,
            y_test,
            y_pred,
            id2label,
            label2id,
            idx_test,
        ) = run_train_rf(re_dataset)

        # ===== LOG PARAM =====
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("phobert_name", PHOBERT_NAME)
        mlflow.log_param(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== EVALUATION =====
        results = full_evaluation(
            y_true=y_test,
            y_pred=y_pred,
            id2label=id2label,
            label2id=label2id,
            samples=re_dataset,
            idx_test=idx_test,
        )

        # ===== LOG MODEL =====
        mlflow.sklearn.log_model(clf, artifact_path="model")

        log_eval_results(
            results,
            metric_prefix="test",
            artifact_dir="evaluation",
        )

        reg_info = register_and_promote(
            run_id=run.info.run_id,
            name="re_rf",
            artifact_path="model",
        )

        print("Run ID:", run.info.run_id)
        print("Registered:", reg_info)


if __name__ == "__main__":
    main()
