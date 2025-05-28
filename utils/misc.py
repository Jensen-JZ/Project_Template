import datetime
import os
import subprocess
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

def send_message(message, title="Notification", platform_name="default", timeout=5):
    """
    Sends a notification message to a remote server or prints to console as fallback.
    
    Supported platforms:
        - default: Local console fallback
        - dingtalk: 钉钉 (DingTalk)
        - wechat: 微信 (WeChat)
        - feishu: 飞书 (Feishu)
        - serverchan: Server 酱 (ServerChan)
        - bark: Bark (iOS)
        - telegram: Telegram
        - pushplus: PushPlus
        - gotify: Gotify

    Args:
        message (str): The message body to be sent.
        title (str): The message title or subject line.
        platform_name (str): Target platform type, one of ["default", "dingtalk", "wechat", "feishu", "serverchan", ...].
        timeout (int): Timeout for the request in seconds (default: 5 seconds).

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """

    base_url = os.environ.get("MESSAGE_PUSH_URL")
    platform_name = platform_name.lower()

    if not base_url:
        print(f"[Fallback] {title}: {message}")
        return False

    try:
        if platform_name == "wechat":
            url = f"{base_url}?type=corp&title={quote(title)}&description={quote(message)}"
            res = requests.get(url, timeout=timeout)

        elif platform_name == "dingtalk":
            payload = {
                "msgtype": "text",
                "text": {"content": f"{title}\n{message}"},
            }
            res = requests.post(base_url, json=payload, timeout=timeout)

        elif platform_name == "feishu":
            payload = {
                "msg_type": "text",
                "content": {"text": f"{title}\n{message}"},
            }
            res = requests.post(base_url, json=payload, timeout=timeout)

        elif platform_name == "serverchan":
            url = f"{base_url}?text={quote(title)}&desp={quote(message)}"
            res = requests.get(url, timeout=timeout)

        elif platform_name == "bark":
            url = f"{base_url}/{quote(title)}/{quote(message)}"
            res = requests.get(url, timeout=timeout)

        elif platform_name == "telegram":
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID")
            if not token or not chat_id:
                raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": f"{title}\n{message}",
            }
            res = requests.post(url, json=payload, timeout=timeout)

        elif platform_name == "pushplus":
            token = os.environ.get("PUSHPLUS_TOKEN")
            if not token:
                raise ValueError("PUSHPLUS_TOKEN not set.")
            url = f"https://www.pushplus.plus/send"
            payload = {
                "token": token,
                "title": title,
                "content": message,
            }
            res = requests.post(url, json=payload, timeout=timeout)

        elif platform_name == "gotify":
            token = os.environ.get("GOTIFY_TOKEN")
            if not token:
                raise ValueError("GOTIFY_TOKEN not set.")
            url = f"{base_url}/message?token={token}"
            payload = {
                "title": title,
                "message": message,
                "priority": 5,
            }
            res = requests.post(url, json=payload, timeout=timeout)

        else:
            print(f"[Local Fallback] {title}: {message}")
            return False

        if res.status_code != 200:
            print(f"[Error] Failed to push message: {res.status_code} - {res.text}")
            return False

        return True

    except Exception as e:
        print(f"[Exception] Failed to send message: {e}")
        return False


def get_datetime(short=False):
    format_str = '%Y%m%d%H%M%S' if short else '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.now().strftime(format_str)


def get_commit_hash():
    process = subprocess.Popen(['git', 'log', '-n', '1'], stdout=subprocess.PIPE)
    output = process.communicate()[0]
    output = output.decode('utf-8')
    return output[7:13]


def start_tensorboard(working_dir, logdir='logs'):
    try:
        process = subprocess.Popen(['tensorboard', '--logdir', logdir, '--bind_all'], cwd=working_dir)
    except FileNotFoundError as e:
        print(f"Error: failed to start Tensorboard -- {e}")


def str2bool(v):
    return v.lower() in ['true']


def str2list(string, separator='-', target_type=int):
    return list(map(target_type, string.split(separator)))
