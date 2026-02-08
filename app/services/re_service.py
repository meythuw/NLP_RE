import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def prepare_entities_for_re(ner_output: dict) -> list[dict]:
    entities = ner_output.get("entities", [])

    prepared = []
    used_ids = set()
    counter = 1

    for ent in entities:
        ent = ent.copy()

        if "id" not in ent:
            new_id = f"e{counter}"
            while new_id in used_ids:
                counter += 1
                new_id = f"e{counter}"
            ent["id"] = new_id
            counter += 1

        used_ids.add(ent["id"])

        ent["text"] = normalize_text(ent["text"])

        for f in ("id", "type", "text", "start", "end"):
            if f not in ent:
                raise ValueError(f"Missing {f} in entity: {ent}")

        prepared.append(ent)

    return prepared
def vectorize(entities: list, text: str):
    """
    Input: 
        - entities: từ prepare_re
        - text: câu gốc
    Output: (X, pairs_info)
        X: numpy array (n_pairs, 3840)
        pairs_info: list metadata từng cặp
    """
    if len(entities) < 2:
        return np.array([]), []
    
    features = []
    pairs_info = []
    
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue
            
            h = entities[i]
            t = entities[j]
            
            # Inject markers
            marked = _inject(text, h, t)
            
            # PhoBERT
            enc = _tokenizer(marked, return_tensors="pt", 
                           truncation=True, max_length=256)
            
            with torch.no_grad():
                out = _model(**enc)
                emb = out.last_hidden_state[0]  # (seq_len, 768)
            
            # Features
            mask = enc["attention_mask"][0].unsqueeze(-1).float()
            sent = (emb * mask).sum(0) / mask.sum()
            
            tokens = _tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            
            def get_vec(marker):
                idx = [k for k, tok in enumerate(tokens) if marker in tok]
                return emb[idx[0]+1:idx[1]].mean(0) if len(idx) >= 2 else sent
            
            h_vec = get_vec("[HEAD]") 
            t_vec = get_vec("[TAIL]")
            diff = torch.abs(h_vec - t_vec)
            mul = h_vec * t_vec
            
            feat = torch.cat([sent, h_vec, t_vec, diff, mul])
            
            features.append(feat.numpy())
            pairs_info.append({
                "head_id": h["id"],
                "head_text": h["text"],
                "head_type": h["type"],
                "tail_id": t["id"],
                "tail_text": t["text"],
                "tail_type": t["type"]
            })
    
    return np.vstack(features) if features else np.array([]), pairs_info

def _inject(text: str, head: dict, tail: dict) -> str:
    """Inject markers [HEAD] và [TAIL]"""
    markers = [
        (head["start"], head["end"], "[HEAD]", "[/HEAD]"),
        (tail["start"], tail["end"], "[TAIL]", "[/TAIL]")
    ]
    markers.sort(key=lambda x: x[0], reverse=True)
    
    result = text
    for start, end, open_m, close_m in markers:
        result = result[:end] + f" {close_m}" + result[end:]
        result = result[:start] + f"{open_m} " + result[start:]
    
    return result
MODEL_NAME = "vinai/phobert-base"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME)
_model.eval()

if __name__ == "__main__":
    ner_output = {
        "text": "Niacinamide giúp giảm mụn hiệu quả",
        "entities": [
            {
                "start": 0,
                "end": 11,
                "text": "Niacinamide",
                "type": "INGREDIENT"
            },
            {
                "start": 17,
                "end": 25,
                "text": "giảm mụn",
                "type": "BENEFIT"
            }
        ]
    }

    # 1. Prepare entities
    prepared = prepare_entities_for_re(ner_output)

    print("=== PREPARED ENTITIES ===")
    for e in prepared:
        print(e)

    # 2. Vectorize
    X, pairs_info = vectorize(prepared, ner_output["text"])

    print("\n=== VECTORIZE RESULT ===")
    print("X shape:", X.shape)

    print("\n=== PAIRS INFO ===")
    for p in pairs_info:
        print(p)
