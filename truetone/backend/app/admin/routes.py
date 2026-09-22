from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import uuid

router = APIRouter()

class SiprecEndpoint(BaseModel):
    id: str
    name: str
    url: str
    status: str = "unknown"

endpoints_db: List[SiprecEndpoint] = []

class RegisterEndpointRequest(BaseModel):
    name: str
    url: str

@router.post("/admin/siprec")
async def register_endpoint(req: RegisterEndpointRequest):
    endpoint = SiprecEndpoint(
        id=uuid.uuid4().hex[:8],
        name=req.name,
        url=req.url,
        status="healthy"  # Mock status
    )
    endpoints_db.append(endpoint)
    return {"status": "ok", "endpoint": endpoint}

@router.get("/admin/siprec", response_model=List[SiprecEndpoint])
async def list_endpoints():
    return endpoints_db

@router.post("/admin/siprec/{endpoint_id}/health")
async def check_health(endpoint_id: str):
    # Mock health check
    for ep in endpoints_db:
        if ep.id == endpoint_id:
            ep.status = "healthy"
            return {"status": "ok", "health": "healthy"}
    return {"status": "error", "message": "Endpoint not found"}
