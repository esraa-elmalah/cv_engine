from typing import List
from pydantic import BaseModel, Field


class CVTemplate(BaseModel):
    """Template for CV generation."""
    role: str
    level: str
    industry: str
    skills: List[str]
    experience_years: int
