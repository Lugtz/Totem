from typing import Any, Dict, List
 
#Campos que queremos mostrar en el HUD, en orden
CAMPOS_DASH = ["nombre", "empresa", "email", "telefono", "solucion"]
 
#Estado en memoria: { session_id: { campo: valor } }
estado_dashboard: Dict[str, Dict[str, Any]] = {}


def actualizar_dashboard_desde_slots(session_id: str, slots: Dict[str, Any]) -> None:
    """
    Actualiza el estado del dashboard con los slots nuevos
    detectados en la conversación para una sesión.
 
    Se llama cada vez que procesas un turno de chat.
    """
    if not isinstance(slots, dict):
        return
 
    ses = estado_dashboard.setdefault(session_id, {})
 
    for campo in CAMPOS_DASH:
        if campo in slots:
            valor = slots.get(campo)
            if valor is None:
                continue
 
            # Normalizamos a string
            if isinstance(valor, str):
                txt = valor.strip()
            else:
                txt = str(valor).strip()
 
            if txt:
                ses[campo] = txt


def construir_estado_dashboard(session_id: str) -> Dict[str, Any]:
    """
    Devuelve el JSON que usará el HUD:
 
    {
      "session_id": "...",
      "fields": [
        {"id": "nombre", "label": "NOMBRE", "value": "LUIS PÉREZ", "filled": true},
        ...
      ],
      "filled_count": 3,
      "missing_count": 2,
      "missing": ["telefono","solucion"]
    }
    """
    ses = estado_dashboard.get(session_id, {})
 
    fields: List[Dict[str, Any]] = []
    filled = 0
 
    for campo in CAMPOS_DASH:
        raw = ses.get(campo)
        value = ""
        if isinstance(raw, str):
            value = raw.strip()
        elif raw is not None:
            value = str(raw).strip()
 
        is_filled = bool(value)
 
        if is_filled:
            filled += 1
 
        # etiqueta bonita
        if campo == "solucion":
            label = "SOLUCIÓN"
        else:
            label = campo.upper()
 
        fields.append(
            {
                "id": campo,
                "label": label,
                "value": value if is_filled else None,
                "filled": is_filled,
            }
        )
 
    missing = [f["id"] for f in fields if not f["filled"]]
 
    return {
        "session_id": session_id,
        "fields": fields,
        "filled_count": filled,
        "missing_count": len(missing),
        "missing": missing,
    }