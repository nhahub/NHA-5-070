from pydantic import BaseModel, Field


class DecisionScores(BaseModel):
    answer: float = Field(ge=0.0, le=1.0)
    ask_to_repeat: float = Field(ge=0.0, le=1.0)
    defer: float = Field(ge=0.0, le=1.0)
    stay_silent: float = Field(ge=0.0, le=1.0)