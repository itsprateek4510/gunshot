from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class SystemMode(str, Enum):
    DEMO = "demo"   # Clap detection
    REAL = "real"   # Gunshot detection


class MicPosition(BaseModel):
    mic_id: str
    lat: float = 28.6139    # GPS latitude
    lng: float = 77.2090    # GPS longitude
    x: float = 0.0          # Local X coordinate (metres)
    y: float = 0.0          # Local Y coordinate (metres)
    name: Optional[str] = None

    class Config:
        extra = "allow"


class TriadConfig(BaseModel):
    triad_id: str
    name: Optional[str] = "Triad 1"
    mics: List[MicPosition] = []

    class Config:
        extra = "allow"
