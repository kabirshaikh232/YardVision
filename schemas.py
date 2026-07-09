from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class DetectionBase(BaseModel):
    class_name: str
    confidence: float
    ocr_text: Optional[str] = None
    zone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class DetectionCreate(DetectionBase):
    pass


class DetectionResponse(DetectionBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class InventoryItem(BaseModel):
    class_name: str
    count: int
    last_seen: datetime
    zone: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    total_detections: int
    active_zones: List[str]