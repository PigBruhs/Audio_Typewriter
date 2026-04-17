from dataclasses import dataclass


@dataclass(slots=True)
class WordOccurrence:
    token: str
    source_audio_id: str
    start_sec: float
    end_sec: float
    confidence: float


@dataclass(slots=True)
class MixPlanItem:
    token: str
    source_audio_id: str
    start_sec: float
    end_sec: float

