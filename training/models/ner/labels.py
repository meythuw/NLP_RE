from typing import Dict, List, Tuple

def build_label_mapping(datasets: List[List[str]]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_set = set()
    for seq in datasets:
        for tag in seq:
            label_set.add(tag)

    label_list = sorted(label_set)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label_list, label2id, id2label
