from pydantic import BaseModel, EmailStr
from typing import List

class Experience(BaseModel):
    title: str
    company: str
    years: str
    description: str

class Education(BaseModel):
    degree: str
    institution: str
    year: str

class CV(BaseModel):
    name: str
    email: EmailStr
    phone: str
    role: str
    summary: str
    skills: List[str]
    experience: List[Experience]
    education: List[Education]
    certifications: List[str]