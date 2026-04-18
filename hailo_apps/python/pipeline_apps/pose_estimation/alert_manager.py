import os
import time
import cv2
import threading
import subprocess
from datetime import datetime
from abc import ABC, abstractmethod

from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

class AlertProvider(ABC):
    @abstractmethod
    def send_alert(self, message: str, image=None):
        pass

class TelegramAlertProvider(AlertProvider):
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def _execute_telegram_alert(self, image_to_send, current_time, message):
        snapshot_path = None
        try:
            if image_to_send is not None:
                time_str = datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')
                snapshot_path = f"/tmp/alert_snapshot_{time_str}.jpg"
                cv2.imwrite(snapshot_path, image_to_send)
                cmd = [
                    "curl", "-s",
                    "-F", f"chat_id={self.chat_id}",
                    "-F", f"photo=@{snapshot_path}",
                    "-F", f"caption={message}",
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                ]
            else:
                cmd = [
                    "curl", "-s",
                    "-d", f"chat_id={self.chat_id}",
                    "--data-urlencode", f"text={message}",
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            hailo_logger.error("Failed to execute Telegram alert: %s", e)
        finally:
            if snapshot_path:
                try:
                    os.remove(snapshot_path)
                except OSError:
                    pass

    def send_alert(self, message: str, image=None):
        image_to_send = image.copy() if image is not None else None
        t = threading.Thread(
            target=self._execute_telegram_alert,
            args=(image_to_send, time.time(), message),
            daemon=True,
        )
        t.start()

class EmailAlertProvider(AlertProvider):
    def __init__(self, email_address: str):
        self.email_address = email_address

    def send_alert(self, message: str, image=None):
        hailo_logger.info(f"Email alert stub: Would send to {self.email_address} with message: {message}")
        # TODO: Implement actual email sending logic

class SMSAlertProvider(AlertProvider):
    def __init__(self, phone_number: str):
        self.phone_number = phone_number

    def send_alert(self, message: str, image=None):
        hailo_logger.info(f"SMS alert stub: Would send to {self.phone_number} with message: {message}")
        # TODO: Implement actual SMS sending logic (e.g., using Twilio)

class AlertManager:
    def __init__(self):
        self.providers = []

    def add_provider(self, provider: AlertProvider):
        self.providers.append(provider)

    def send_alert(self, message: str, image=None):
        for provider in self.providers:
            provider.send_alert(message, image)
