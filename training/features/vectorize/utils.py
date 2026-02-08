from training.features.build_data.build_re_dataset import build_re_datasets

# python -m training.features.svm_re.utils
HEAD_S, HEAD_E = "[HEAD]", "[/HEAD]"
TAIL_S, TAIL_E = "[TAIL]", "[/TAIL]"

def build_label_maps(samples):
    labels = sorted({s["relation"] for s in samples})
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in id2label.items()}
    return id2label, label2id


def span_to_tuple(span) -> tuple[int, int]:
    if isinstance(span, dict):
        return int(span["start"]), int(span["end"])
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return int(span[0]), int(span[1])
    raise TypeError(f"Unsupported span format: {type(span)} -> {span}")

#cell 93
def soft_match(a: str, b: str) -> bool:
    na = " ".join(a.strip().split())
    nb = " ".join(b.strip().split())
    return na == nb


def inject_markers_segmented(sample: dict):
    sep = " [SEP] "

    # Decide segments/ khúc này thay = cell 127 nha, m lấy hàm đó đưa ra ngoài 
    if "sentence_a" in sample and "sentence_b" in sample:
        sent_a = sample["sentence_a"]
        sent_b = sample["sentence_b"]
        base_a, base_b = sent_a, sent_b
    else:
        base_a = sample["sentence"]
        base_b = ""  # no segment B

    # Extract spans + seg
    hspan = sample["head_span"]
    tspan = sample["tail_span"]

    #Cell 83, 94 bên kia, cân nhắc xem có cần sửa gì không nha
    hseg = hspan.get("seg", "A") if isinstance(hspan, dict) else "A"
    tseg = tspan.get("seg", "A") if isinstance(tspan, dict) else "A"

    hs, he = span_to_tuple(hspan)
    ts, te = span_to_tuple(tspan)

    # Helper: inject into one segment only
    #hàm này t sửa trong file notebook có gì xem thử nha 
    def inject_one(sentence: str, s: int, e: int, S: str, E: str, expected_text: str | None = None):
        if sentence is None:
            raise ValueError("Missing sentence segment")
        if not (0 <= s < e <= len(sentence)):
            raise ValueError(f"Span out of range for segment: {s}-{e} len={len(sentence)}")

        actual = sentence[s:e]
        if expected_text is not None and not soft_match(actual, expected_text):
            print(f"[WARN] span mismatch: '{actual}' != '{expected_text}'")

        return sentence[:s] + f"{S} " + sentence[s:e] + f" {E}" + sentence[e:]

    #khúc này sửa lại cách viết cho nó tường minh nha, t thấy nó hơi gộp
    # Inject (reverse order per segment to keep indices stable)
    a = base_a
    b = base_b

    # Collect per segment edits then apply from right to left
    edits_a = []
    edits_b = []

    if hseg == "A": edits_a.append(("HEAD", hs, he, HEAD_S, HEAD_E, sample.get("head_text")))
    else:          edits_b.append(("HEAD", hs, he, HEAD_S, HEAD_E, sample.get("head_text")))

    if tseg == "A": edits_a.append(("TAIL", ts, te, TAIL_S, TAIL_E, sample.get("tail_text")))
    else:           edits_b.append(("TAIL", ts, te, TAIL_S, TAIL_E, sample.get("tail_text")))

    # Apply edits in each segment from right to left
    edits_a.sort(key=lambda x: x[1], reverse=True)
    edits_b.sort(key=lambda x: x[1], reverse=True)

    for _, s, e, S, E, exp in edits_a:
        a = inject_one(a, s, e, S, E, exp)
    for _, s, e, S, E, exp in edits_b:
        b = inject_one(b, s, e, S, E, exp)

    # Build final input string
    if b:
        return a.strip() + sep + b.strip()
    return a