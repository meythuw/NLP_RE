from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

EntityType = Literal["NAME", "INCI", "ORIGIN", "BENEFITS", "SKIN_CONCERNS"]
RelationType = Literal[
    "has_inci_name",
    "has_origin",
    "has_benefits",
    "targets_skin_concerns",
    "no_relation",
]

class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1)

class NormalizeResponse(BaseModel):
    normalized_text: str

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)

class Entity(BaseModel):
    id: str = Field(..., min_length=1, description="id của entity")
    type: EntityType
    text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    conf: Optional[float] = Field(None, ge=0.0, le=1.0)

class NERResponse(BaseModel):
    entities: List[Entity]

class EntitiesValidateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    entities: List[Entity]

class ValidationIssue(BaseModel):
    code: str
    message: str
    field: Optional[str] = None
    entity_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class EntitiesValidateResponse(BaseModel):
    ok: bool
    issues: List[ValidationIssue] = []

class Pair(BaseModel):
    head_id: str = Field(..., min_length=1)
    tail_id: str = Field(..., min_length=1)

class PairsRequest(BaseModel):
    entities: List[Entity]
    allow_same_span: bool = False

class PairsResponse(BaseModel):
    pairs: List[Pair]

class REPredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    re_model: Literal["svm", "logreg", "rf", "phobert"]
    entities: List[Entity]
    pairs: List[Pair]
    threshold: float = Field(0.0, ge=0.0, le=1.0)

class Relation(BaseModel):
    head_id: str
    tail_id: str
    relation: RelationType
    confidence: float = Field(..., ge=0.0, le=1.0)

class REPredictResponse(BaseModel):
    relations: List[Relation]
    meta: Dict[str, Any] = {}
