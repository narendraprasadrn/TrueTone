from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.alerts.dashboard_ws import manager

ws_router = APIRouter()

@ws_router.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect messages from client for now, just keep alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
