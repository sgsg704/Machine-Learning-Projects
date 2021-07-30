from pydantic import BaseModel


class BankNote(BaseModel):
    variance: float
    skewness: float
    kurtosis: float
    entropy: float