import numpy as np
import torch
from training.features.vectorize.utils import inject_markers_segmented
from transformers import AutoTokenizer, AutoModel
from training.features.build_data.build_re_dataset import build_re_datasets

HEAD_S, HEAD_E = "[HEAD]", "[/HEAD]"
TAIL_S, TAIL_E = "[TAIL]", "[/TAIL]"

@torch.no_grad()
def phobert_vectorize_re_sample_tensor(
    sample: dict,
    tokenizer,
    model,
    device: str = "cpu",
    max_len: int = 256,
) -> torch.Tensor:
    # 1) Inject marker
    marked_text = inject_markers_segmented(sample)

    # 2) Tokenize + move to device
    encoding = tokenizer(
        marked_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 3) Run PhoBERT
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_size)

    # 4) ids -> tokens to find marker indices
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    def find_marker_index(marker: str) -> int | None:
        for i, tok in enumerate(tokens):
            if marker in tok:
                return i
        return None

    head_start = find_marker_index(HEAD_S)
    head_end   = find_marker_index(HEAD_E)
    tail_start = find_marker_index(TAIL_S)
    tail_end   = find_marker_index(TAIL_E)

    mask = attention_mask[0].unsqueeze(-1).to(token_embeddings.dtype)  # (seq_len, 1)
    denom = mask.sum().clamp_min(1.0)
    sent_vector = (token_embeddings * mask).sum(dim=0) / denom          # (hidden,)

    # 6) Entity span pooling
    def mean_pool_span(start: int | None, end: int | None) -> torch.Tensor | None:
        if start is None or end is None:
            return None
        if end <= start + 1:
            return None
        return token_embeddings[start + 1 : end].mean(dim=0)

    head_vector = mean_pool_span(head_start, head_end) or sent_vector
    tail_vector = mean_pool_span(tail_start, tail_end) or sent_vector

    #7. Interaction features
    diff_vector = torch.abs(head_vector - tail_vector)
    mul_vector  = head_vector * tail_vector

    # 8) Final feature
    feature = torch.cat(
        [sent_vector, head_vector, tail_vector, diff_vector, mul_vector],
        dim=0,
    )  # (5*hidden,)

    return feature.detach().cpu()


def build_xy_ml(samples, tokenizer, model, device="cpu", max_len=256, label2id=None):
    """Trả về numpy array để huấn luyện các mô hình ML truyền thống (SVM, LR, RF)."""
    X_list, y_list, idx_list = [], [], []
    bad = 0

    for i, ex in enumerate(samples):
        try:
            feat = phobert_vectorize_re_sample_tensor(
                ex, tokenizer, model, device=device, max_len=max_len
            )  # torch.Tensor (cpu)
            X_list.append(feat.numpy())  # -> np
            y_list.append(ex["relation"] if label2id is None else label2id[ex["relation"]])
            idx_list.append(i)
        except ValueError as e:
            bad += 1
            print(f"[SKIP] {e} | relation={ex.get('relation')} | task_id={ex.get('task_id')}")

    print(f"Skipped {bad} bad samples")
    return np.vstack(X_list), np.array(y_list), np.array(idx_list)


def build_xy_dl(samples, tokenizer, model, device="cpu", max_len=256, label2id=None):
    """Trả về tensor PyTorch để huấn luyện các mô hình deep learning."""
    X_list, y_list, idx_list = [], [], []
    bad = 0

    for i, ex in enumerate(samples):
        try:
            feat = phobert_vectorize_re_sample_tensor(
                ex, tokenizer, model, device=device, max_len=max_len
            )  # torch.Tensor (cpu)
            X_list.append(feat)  # keep torch
            y_list.append(ex["relation"] if label2id is None else label2id[ex["relation"]])
            idx_list.append(i)
        except ValueError as e:
            bad += 1
            print(f"[SKIP] {e} | relation={ex.get('relation')} | task_id={ex.get('task_id')}")

    print(f"Skipped {bad} bad samples")

    X = torch.stack(X_list, dim=0)  # (N, feat_dim)

    # y: nếu label2id đã map ra int -> tensor long; nếu chưa map -> bạn cần map trước (DL thường cần int)
    if len(y_list) > 0 and isinstance(y_list[0], str):
        raise ValueError("build_xy_dl cần label2id để map relation (str) -> int (class id).")

    y = torch.tensor(y_list, dtype=torch.long)      # (N,)
    idx = torch.tensor(idx_list, dtype=torch.long)  # (N,)
    return X, y, idx

# ##       CHẠY VỚI THUẬT TOÁN MACHINE LEARNING
# if __name__ == "__main__":
#     # Load dataset
#     dataset_re = build_re_datasets()
#     samples = dataset_re[:5]
#     print("Số sample test:", len(samples))
#     print("Ví dụ 1 sample:")
#     print(samples[0])

#     # Load PhoBERT (CPU)
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#     model = AutoModel.from_pretrained("vinai/phobert-base").to("cpu")
#     model.eval()

#     # Label mapping
#     labels = sorted(set(s["relation"] for s in samples))
#     label2id = {l: i for i, l in enumerate(labels)}
#     print("Label2ID:", label2id)
   
#    # Vectorize → X, y
#     X, y, idx = build_xy_ml(
#         samples,
#         tokenizer,
#         model,
#         device="cpu",
#         label2id=label2id
#     )
#     # In kết quả
#     print("X shape:", X.shape)   # (N, 3840)
#     print("y shape:", y.shape)
#     print("y:", y)
#     print("idx:", idx)
#     print("Vector sample (5 số đầu):", X[0][:5])



# ##     CHẠY VỚI THUẬT TOÁN DEEP LEARNING
# if __name__ == "__main__":

#     # Load dataset
#     dataset_re = build_re_datasets()
#     samples = dataset_re[:5]

#     print("Số sample test:", len(samples))
#     print("Ví dụ 1 sample:")
#     print(samples[0])

#     # Load PhoBERT (CPU)
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#     model = AutoModel.from_pretrained("vinai/phobert-base")
#     model.eval()

#     # Label mapping (BẮT BUỘC cho DL)
#     labels = sorted(set(s["relation"] for s in samples))
#     label2id = {l: i for i, l in enumerate(labels)}
#     print("Label2ID:", label2id)

#     # Vectorize → DL
#     X, y, idx = build_xy_dl(
#         samples,
#         tokenizer,
#         model,
#         device="cpu",
#         label2id=label2id
#     )

#     print("X shape:", X.shape)   # (N, 3840)
#     print("y shape:", y.shape)
#     print("y:", y)
#     print("idx:", idx)
#     print("Vector sample (5 số đầu):", X[0][:5])
