import os
from training.features.build_data.build_ner_dataset import build_ner_dataset
from training.models.ner.config import CRFConfig
from training.models.ner.data_io import read_jsonl, to_xy
from training.models.ner.features import build_features
from training.models.ner.model import train_crf, save_model, load_model, predict
from training.models.ner.evaluate import evaluate_seqeval, print_report
from training.models.ner.infer import decode_entities, pretty_print

def simple_tokenize(sentence: str):
    # Placeholder: bạn nên dùng đúng tokenizer lúc gán nhãn trong Label Studio
    return sentence.strip().split()

def main():
    cfg = CRFConfig(
        window=2,
        c1=0.1,
        c2=0.1,
        max_iterations=200,
    )
    train_data, valid_data = build_ner_dataset()

    X_train_tokens, y_train = to_xy(train_data)
    X_valid_tokens, y_valid = to_xy(valid_data)

    # 2) Build features
    X_train_feats = build_features(X_train_tokens, window=cfg.window)
    X_valid_feats = build_features(X_valid_tokens, window=cfg.window)

    # 3) Train
    crf = train_crf(X_train_feats, y_train, cfg)

    # 4) Evaluate
    y_pred = predict(crf, X_valid_feats)
    metrics = evaluate_seqeval(y_valid, y_pred)
    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\n=== SeqEval report ===")
    print_report(y_valid, y_pred)


    # 6) Quick test 1 câu
    test_sentence = "Vitamin C giúp làm sáng da và giảm mụn hiệu quả."
    tokens = simple_tokenize(test_sentence)
    feats = build_features([tokens], window=cfg.window)
    tags = predict(crf, feats)[0]

    print("\n=== Token predictions ===")
    for t, tag in zip(tokens, tags):
        print(f"{t:15s} -> {tag}")

    ents = decode_entities(tokens, tags)
    print("\n=== Decoded entities ===")
    pretty_print(ents)

if __name__ == "__main__":
    main()
