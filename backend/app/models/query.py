from pydantic import BaseModel

class GenerateCVRequest(BaseModel):
    count: int = 25