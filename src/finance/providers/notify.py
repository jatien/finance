"""Canals d'alertes: ntfy (recomanat: app mòbil, sense compte), Telegram i correu (SMTP).

Cada canal implementa `Notifier.send` i llança `ProviderError` si el missatge no s'ha lliurat;
el job d'alertes prova tots els canals configurats i registra quins han funcionat.
"""

from __future__ import annotations

import contextlib
import smtplib
from email.message import EmailMessage
from urllib.parse import urlsplit

import requests

from finance.providers.base import Notifier, ProviderError
from finance.settings import Settings

LEVELS = ("info", "important", "critical")
TIMEOUT = 15


def level_rank(level: str) -> int:
    return LEVELS.index(level) if level in LEVELS else 0


class NtfyNotifier(Notifier):
    """Publica a un tema de ntfy (ntfy.sh o servidor propi) amb l'API JSON (admet UTF-8)."""

    name = "ntfy"
    PRIORITY = {"info": 3, "important": 4, "critical": 5}
    TAGS = {"info": ["chart_with_upwards_trend"], "important": ["bell"], "critical": ["rotating_light"]}

    def __init__(self, url: str, token: str | None = None):
        parts = urlsplit(url.rstrip("/"))
        topic = parts.path.strip("/")
        if not parts.scheme or not topic or "/" in topic:
            raise ValueError(f"URL de ntfy no vàlida (cal https://servidor/tema): {url}")
        self.server = f"{parts.scheme}://{parts.netloc}"
        self.topic = topic
        self.token = token

    def send(self, title: str, body: str, level: str = "info") -> None:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {"topic": self.topic, "title": title[:250], "message": body[:3900],
                   "priority": self.PRIORITY.get(level, 3), "tags": self.TAGS.get(level, []),
                   "markdown": True}
        try:
            r = requests.post(self.server, json=payload, headers=headers, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise ProviderError(f"ntfy: {e}") from e
        if r.status_code >= 300:
            raise ProviderError(f"ntfy: HTTP {r.status_code} {r.text[:200]}")


class TelegramNotifier(Notifier):
    """Bot de Telegram (BotFather) → xat propi. Text pla: evita problemes d'escapament."""

    name = "telegram"
    ICON = {"info": "ℹ️", "important": "🔔", "critical": "🚨"}

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def send(self, title: str, body: str, level: str = "info") -> None:
        text = f"{self.ICON.get(level, '')} {title}\n\n{body}"[:4000]
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text,
                                    "disable_web_page_preview": True}, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise ProviderError("telegram: error de xarxa") from e  # el token va a la URL: no el mostris
        if r.status_code >= 300:
            raise ProviderError(f"telegram: HTTP {r.status_code}")


class EmailNotifier(Notifier):
    name = "email"

    def __init__(self, host: str, port: int, user: str | None, password: str | None,
                 sender: str, to: str):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.to = sender, to

    def send(self, title: str, body: str, level: str = "info") -> None:
        msg = EmailMessage()
        msg["Subject"] = f"[finance{'' if level == 'info' else ' · ' + level}] {title}"
        msg["From"] = self.sender
        msg["To"] = self.to
        msg.set_content(body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=TIMEOUT) as s:
                s.starttls()
                if self.user:
                    s.login(self.user, self.password or "")
                s.send_message(msg)
        except (OSError, smtplib.SMTPException) as e:
            raise ProviderError(f"correu: {type(e).__name__}: {e}") from e


def build_notifiers(settings: Settings) -> list[Notifier]:
    out: list[Notifier] = []
    if settings.alert_ntfy_url:
        out.append(NtfyNotifier(settings.alert_ntfy_url, settings.alert_ntfy_token))
    if settings.alert_telegram_token and settings.alert_telegram_chat_id:
        out.append(TelegramNotifier(settings.alert_telegram_token, settings.alert_telegram_chat_id))
    if settings.alert_smtp_host and settings.alert_smtp_to:
        out.append(EmailNotifier(settings.alert_smtp_host, settings.alert_smtp_port, settings.alert_smtp_user,
                                 settings.alert_smtp_password,
                                 settings.alert_smtp_from or settings.alert_smtp_user or settings.alert_smtp_to,
                                 settings.alert_smtp_to))
    return out


def ping(url: str, suffix: str = "") -> None:
    """Ping a un servei de "dead man's switch" (healthchecks.io): si no arriba, avisa ell."""
    with contextlib.suppress(requests.RequestException):
        requests.get(url.rstrip("/") + suffix, timeout=TIMEOUT)
