import os, requests, time
from core.logger import get_logger

LOG = get_logger(__name__)
FLOW_URL = os.getenv("FLOW_URL")
WRITER_DOC_ID = os.getenv("WRITER_DOC_ID")

def trigger(slots: dict) -> str | None:
    if not FLOW_URL:
        LOG.error({"evt":"flow_missing"})
        return None
    payload = {"slots": slots, "writer_doc_id": WRITER_DOC_ID, "send_email": False, "channel": "stand_ia3"}
    for attempt in range(1, 4):
        try:
            r = requests.post(FLOW_URL, json=payload, timeout=15)
            try:
                body = r.json()
            except Exception:
                body = {}
            if 200 <= r.status_code < 300:
                return body.get("proposalPdfUrl") or body.get("pdf_url") or body.get("url")
        except Exception as e:
            LOG.warning({"evt":"flow_err","attempt":attempt,"err":str(e)})
            time.sleep(0.5 * attempt)
    return None
