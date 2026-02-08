import re
from typing import Dict, List

_WORD_RE = re.compile(r"^\w+$", re.UNICODE)

def word_shape(w: str) -> str:
    out = []
    for ch in w:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)

def token_features(tokens: List[str], i: int) -> Dict[str, object]:
    w = tokens[i]
    feats: Dict[str, object] = {
        "bias": 1.0,
        "w": w,
        "lower": w.lower(),
        "is_upper": w.isupper(),
        "is_title": w.istitle(),
        "is_digit": w.isdigit(),
        "has_digit": any(ch.isdigit() for ch in w),
        "has_hyphen": "-" in w,
        "shape": word_shape(w),
        "len": len(w),
        "prefix1": w[:1],
        "prefix2": w[:2],
        "prefix3": w[:3],
        "suffix1": w[-1:],
        "suffix2": w[-2:],
        "suffix3": w[-3:],
        "is_word": bool(_WORD_RE.match(w)),
    }
    return feats

def sent2features(tokens: List[str], window: int = 2) -> List[Dict[str, object]]:
    n = len(tokens)
    out = []
    for i in range(n):
        feats = token_features(tokens, i)

        # BOS/EOS flags
        if i == 0:
            feats["BOS"] = True
        if i == n - 1:
            feats["EOS"] = True

        # context window
        for k in range(1, window + 1):
            if i - k >= 0:
                f = token_features(tokens, i - k)
                for key, val in f.items():
                    feats[f"-{k}:{key}"] = val
            else:
                feats[f"-{k}:PAD"] = True

            if i + k < n:
                f = token_features(tokens, i + k)
                for key, val in f.items():
                    feats[f"+{k}:{key}"] = val
            else:
                feats[f"+{k}:PAD"] = True

        out.append(feats)
    return out

def build_features(X_tokens: List[List[str]], window: int = 2) -> List[List[Dict[str, object]]]:
    return [sent2features(tokens, window=window) for tokens in X_tokens]
