import os, smtplib, mimetypes
from email.message import EmailMessage
from core.logger import get_logger

LOG = get_logger(__name__)
SMTP_HOST = os.getenv("SMTP_HOST","smtp.zoho.com")
SMTP_PORT = int(os.getenv("SMTP_PORT","587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@evolucioni3.com")
APP_NAME  = os.getenv("APP_NAME","Evolución i3 – Stand IA³")

def _attach(msg: EmailMessage, path: str):
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    with open(path, "rb") as f:
        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))

def send_infographic(to_email: str, png_path: str, pdf_path: str | None = None):
    if not (SMTP_USER and SMTP_PASS and SMTP_FROM):
        LOG.error({"evt":"smtp_missing"})
        return False
    msg = EmailMessage()
    msg["Subject"] = f"{APP_NAME} – Tu infografía"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content("¡Gracias por visitarnos!\nAdjuntamos tu infografía.\n\nSaludos,\nEvolución i3")
    _attach(msg, png_path)
    if pdf_path and os.path.exists(pdf_path):
        _attach(msg, pdf_path)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
    LOG.info({"evt":"mail_sent","to":to_email})
    return True
