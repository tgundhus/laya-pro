"""The HTTP payload's required fields and optional Laya extensions.

Responses remain ordinary dictionaries. Unknown JSON fields are preserved.
"""
from typing import Dict, List, Literal, TypedDict, Union

JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]
State = Union[str, List[JsonValue], Dict[str, JsonValue]]
MinConfidenceKey = Literal["default", "choice:2", "choice:3-5", "choice:6-10", "choice:11+",
                           "score:2", "score:3-5", "score:6-10", "score:11+",
                           "noul:2", "noul:3-5", "noul:6-10", "noul:11+"]
MinConfidence = Union[float, Dict[MinConfidenceKey, float]]


class _QuestionOptions(TypedDict, total=False):
    option_order: List[int]


class ChoiceQuestion(_QuestionOptions):
    type: Literal["choice"]
    instructions: JsonValue
    criteria: Union[List[str], Dict[str, JsonValue]]


class ScoreQuestion(_QuestionOptions):
    type: Literal["score"]
    instructions: JsonValue
    criteria: List[JsonValue]


class NoulLabels(TypedDict):
    false: str
    true: str


class _NoulQuestionOptions(_QuestionOptions, total=False):
    criteria: Union[None, Dict[Literal["false", "true"], JsonValue]]
    labels: Union[None, NoulLabels]


class NoulQuestion(_NoulQuestionOptions):
    type: Literal["noul"]
    instructions: JsonValue


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]
Questions = Dict[str, Question]


class Action(TypedDict):
    act_probability: float


class Tournament(TypedDict):
    finalists: List[str]
    rounds: int


class _AnswerExtensions(TypedDict, total=False):
    answer_confidence: float
    low_confidence: Literal[True]
    abstention: Literal["passed", "abstained", "unevaluated"]
    abstention_threshold: float
    action: Action


class _ConfidenceExtensions(_AnswerExtensions, total=False):
    x_jev_confidence: float


class _ConfidenceAnswer(_ConfidenceExtensions):
    confidence: float


class _ChoiceExtensions(_ConfidenceAnswer, total=False):
    x_tournament: Tournament


class ChoiceAnswer(_ChoiceExtensions):
    type: Literal["choice"]
    choice: str
    probabilities: Dict[str, float]


class ScoreAnswer(_ConfidenceAnswer):
    type: Literal["score"]
    score: float
    probabilities: Dict[str, float]
    legend: Dict[str, JsonValue]


class _NoulExtensions(_AnswerExtensions, total=False):
    confidence: float


class NoulAnswer(_NoulExtensions):
    type: Literal["noul"]
    noul: float


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class OptionCollapse(TypedDict):
    total: int
    distinct: int
    tokens_per_option: Union[None, int]


class Usage(TokenUsage, total=False):
    state_tokens: int
    state_tokens_dropped: int
    truncated: bool
    truncated_questions: List[str]
    options: Dict[str, OptionCollapse]


class _DetectionExtensions(TypedDict, total=False):
    mixed_segment: Union[None, str]


class LanguageDetection(_DetectionExtensions):
    script: str
    script_profile: Dict[str, float]
    language: Union[None, str]
    is_english: bool
    language_undecided: bool
    diacritic_rate: float
    non_latin_fraction: float


class RouteDecision(TypedDict):
    model: str
    repo: str
    reason: str
    detection: Union[None, LanguageDetection]
    workflow: Union[None, str]


class _PredictionExtensions(TypedDict, total=False):
    routing: RouteDecision


class Prediction(_PredictionExtensions):
    model: str
    answers: Dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]]
    usage: Usage


class BatchPrediction(TypedDict):
    results: List[Prediction]
    total_usage: TokenUsage


class _HealthExtensions(TypedDict, total=False):
    loaded: List[str]
    device: str


class Health(_HealthExtensions):
    status: Literal["ok"]
