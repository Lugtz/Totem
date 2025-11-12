# core/proposal_trigger.py
import os, json, requests
from core.logger import get_logger

LOG = get_logger(__name__)

CATALYST_URL = (
    "https://ai3-897327580.development.catalystserverless.com/server/ai_3_function/open-proposal-fields"
)

def trigger(slots: dict):
    try:
        payload = {"payload": slots}
        LOG.info({"evt": "proposal_trigger_start", "payload": payload})

        res = requests.post(CATALYST_URL, json=payload, timeout=60)
        LOG.info({"evt": "proposal_trigger_response", "status": res.status_code})

        if res.status_code != 200:
            LOG.error({"evt": "proposal_trigger_http_error", "body": res.text[:300]})
            return None

        data = res.json()
        if not data.get("ok"):
            LOG.error({"evt": "proposal_trigger_ai_error", "data": data})
            return None

        # Simulación: guardar el JSON generado en un archivo temporal
        out_path = "data/outputs/last_proposal_fields.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data["fields"], f, ensure_ascii=False, indent=2)

        LOG.info({"evt": "proposal_trigger_success", "out": out_path})
        return out_path  # Devolvemos la ruta del JSON generado localmente
    except Exception as e:
        LOG.error({"evt": "proposal_trigger_error", "error": str(e)})
        return None
