import os
from dotenv import load_dotenv
import mlflow
import json
#python -m app.services.train.mlflow_model

#load model để dùng
def load_sklearn_from_mlflow(model_name: str, model_version: str):
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model_uri = f"models:/{model_name}/{model_version}"
    clf = mlflow.sklearn.load_model(model_uri)
    return clf


def load_pyfunc_from_mlflow(model_name: str, model_version: str | None = None,):
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
    else:
        raise ValueError("Either model_version must be provided")
    model = mlflow.pyfunc.load_model(model_uri)
    return model

#lấy nhãn để lúc ra kết qủa có thể map để ra kết quả 
def load_label_mappings_from_registry(model_name: str, model_version: str):
    model_uri = f"models:/{model_name}/{model_version}"
    artifact_dir = mlflow.artifacts.download_artifacts(
        artifact_uri= model_uri+ "/labels"
    )

    with open(os.path.join(artifact_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    with open(os.path.join(artifact_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)

    return id2label, label2id

# # print(_load_sklearn_from_mlflow(model_name='re_svm2', model_version=1 ))
# print(load_label_mappings_from_registry(model_name='re_svm2', model_version=1))