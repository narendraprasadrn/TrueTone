import logging
import asyncio
import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def _send_webhook_with_retry(url: str, payload: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        for attempt in range(2):
            try:
                response = await client.post(url, json=payload, timeout=5.0)
                response.raise_for_status()
                logger.info(f"Webhook dispatched to {url} successfully.")
                return
            except httpx.RequestError as e:
                logger.warning(f"Webhook request failed (attempt {attempt + 1}/2): {e}")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Webhook HTTP error (attempt {attempt + 1}/2): {e}")
                
            if attempt == 0:
                await asyncio.sleep(1.0)
                
        logger.error(f"Webhook dispatch to {url} ultimately failed after 2 attempts.")

def dispatch_alert_webhook(payload: Dict[str, Any], url: str = "http://localhost:8000/webhook/receive"):
    """
    Fires the webhook asynchronously without blocking the caller.
    """
    asyncio.create_task(_send_webhook_with_retry(url, payload))
