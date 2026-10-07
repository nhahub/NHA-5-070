from pydantic import BaseModel
from typing import Literal


class TwinReply(BaseModel):
    addressed_to_me: bool
    action: Literal[
        "answer",
        "ask_to_repeat",
        "defer",
        "stay_silent",
    ]
    language: Literal[
        "ar",
        "en",
        "mixed",
    ]
    reply_text: str
    confidence: float
    tools_used: list[str]
    sources: list[str]