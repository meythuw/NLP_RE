from __future__ import annotations
from typing import List 
from app.schema import Entity, ValidationIssue

def _normalize_ws(s: str) -> str:
    return " ".join(s.strip().split())


def validate_entities(text: str, entities: List[Entity]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    n = len(text)

    seen = {}
    for e in entities:
        if e.id in seen:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_ID",
                    message=f"Duplicate entity id: {e.id}",
                    field="id",
                    entity_id=e.id,
                )
            )
        seen[e.id] = True

    for e in entities:
        if e.start >= e.end:
            issues.append(
                ValidationIssue(
                    code="BAD_SPAN_ORDER",
                    message="start must be < end",
                    field="start/end",
                    entity_id=e.id,
                    extra={"start": e.start, "end": e.end},
                )
            )
            continue

        if e.start < 0 or e.end > n:
            issues.append(
                ValidationIssue(
                    code="SPAN_OUT_OF_RANGE",
                    message="Span is outside text range",
                    field="start/end",
                    entity_id=e.id,
                    extra={"text_len": n, "start": e.start, "end": e.end},
                )
            )
            continue

        substr = text[e.start:e.end]
        if _normalize_ws(substr) != _normalize_ws(e.text):
            issues.append(
                ValidationIssue(
                    code="TEXT_MISMATCH",
                    message="Entity.text does not match text[start:end] (after whitespace normalization).",
                    field="text",
                    entity_id=e.id,
                    extra={"expected": substr, "got": e.text},
                )
            )

    return issues


def validate_pairs(entities: List[Entity], pairs) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    ids = {e.id for e in entities}

    for i, p in enumerate(pairs):
        if p.head_id not in ids:
            issues.append(
                ValidationIssue(
                    code="PAIR_HEAD_NOT_FOUND",
                    message=f"pairs[{i}].head_id not found in entities",
                    field=f"pairs[{i}].head_id",
                    extra={"head_id": p.head_id},
                )
            )
        if p.tail_id not in ids:
            issues.append(
                ValidationIssue(
                    code="PAIR_TAIL_NOT_FOUND",
                    message=f"pairs[{i}].tail_id not found in entities",
                    field=f"pairs[{i}].tail_id",
                    extra={"tail_id": p.tail_id},
                )
            )
    
        if p.head_id == p.tail_id:
            issues.append(
                ValidationIssue(
                    code="PAIR_SELF_LOOP",
                    message=f"pairs[{i}] head_id == tail_id (self loop)",
                    field=f"pairs[{i}]",
                    extra={"id": p.head_id},
                )
            )

    return issues
