from typing import Dict, List

def decode_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
    """
    tokens: ["Vitamin", "C", "giúp", ...]
    tags:   ["B-NAME", "I-NAME", "O", ...]
    Return: [{"type": "NAME", "text": "Vitamin C"}, ...]
    """
    entities = []
    cur_type = None
    cur_tokens: List[str] = []

    def flush():
        nonlocal cur_type, cur_tokens
        if cur_type and cur_tokens:
            entities.append({"type": cur_type, "text": " ".join(cur_tokens).strip()})
        cur_type = None
        cur_tokens = []

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            flush()
            cur_type = tag[2:]
            cur_tokens = [tok]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_tokens.append(tok)
        else:
            flush()

    flush()
    return entities

def pretty_print(entities: List[Dict[str, str]]):
    for e in entities:
        print(f"{e['type']:12s} | {e['text']}")
