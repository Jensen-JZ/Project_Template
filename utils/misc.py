import datetime
import importlib.util
import os
import subprocess
import sys
from urllib.parse import quote

import requests


def send_message(message, title="Notification", platform="default", timeout=5):
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
        platform (str): Target platform type, one of ["default", "dingtalk", "wechat", "feishu", "serverchan", ...].
        timeout (int): Timeout for the request in seconds (default: 5 seconds).

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """

    base_url = os.environ.get("MESSAGE_PUSH_URL")
    platform = platform.lower()

    if not base_url:
        print(f"[Fallback] {title}: {message}")
        return False

    try:
        if platform == "wechat":
            url = f"{base_url}?type=corp&title={quote(title)}&description={quote(message)}"
            res = requests.get(url, timeout=timeout)

        elif platform == "dingtalk":
            payload = {
                "msgtype": "text",
                "text": {"content": f"{title}\n{message}"},
            }
            res = requests.post(base_url, json=payload, timeout=timeout)

        elif platform == "feishu":
            payload = {
                "msg_type": "text",
                "content": {"text": f"{title}\n{message}"},
            }
            res = requests.post(base_url, json=payload, timeout=timeout)

        elif platform == "serverchan":
            url = f"{base_url}?text={quote(title)}&desp={quote(message)}"
            res = requests.get(url, timeout=timeout)

        elif platform == "bark":
            url = f"{base_url}/{quote(title)}/{quote(message)}"
            res = requests.get(url, timeout=timeout)

        elif platform == "telegram":
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

        elif platform == "pushplus":
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

        elif platform == "gotify":
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


def get_current_time(short=False):
    """
    Returns the current time formatted as a string.

    Args:
        short (bool): If True, returns compact format (e.g., '20250504181233'); otherwise, returns full format.

    Returns:
        str: Current time formatted as a string.
    """
    format_date = "%Y%m%d%H%M%S" if short else "%Y-%m-%d_%H-%M-%S"
    return datetime.datetime.now().strftime(format_date)


def get_commit_hash():
    """
    Returns the short commit hash (first 7 characters) of the latest commit in the current Git repository.

    Returns:
        str: Short commit hash of the latest commit.
    """

    # Launch a subprocess to execute 'git log -n 1' and capture its standard output
    external_command = subprocess.Popen(
        ["git", "log", "-n", "1"], stdout=subprocess.PIPE
    )
    # Wait for the subprocess to finish and retrieve its output (stdout)
    command_output = external_command.communicate()[0]
    commit_hash = command_output.decode("utf-8").split()[1]
    return commit_hash[:7]


def check_tensorboard_installed():
    """
    Checks if TensorBoard is installed in the currrent Python environment.

    Returns:
        bool: True if TensorBoard is installed, False otherwise.
    """
    return importlib.util.find_spec("tensorboard") is not None


def start_tensorboard(working_dir, log_dir="logs", port=6006):
    """
    Launches TensorBoard in the background using subprocess.

    Args:
        working_dir (str): Directory where the TensorBoard server will be started.
        log_dir (str): Directory containing TensorBoard logs (default: 'logs').
        port (int): Port number for TensorBoard server (default: 6006).

    Returns:
        None
    """

    if not check_tensorboard_installed():
        print(
            "TensorBoard is not installed. Please install it using `pip install tensorboard`."
        )
        return

    try:
        external_command = [
            sys.executable,
            "-m",
            "tensorboard",
            "--logdir",
            log_dir,
            "--port",
            str(port),
            "--bind_all",
        ]
        subprocess.Popen(external_command, cwd=working_dir)
        print(f"TensorBoard started at http://localhost:{port}")

    except FileNotFoundError as e:
        print(f"[Error] TensorBoard not found or failed to start: {e}")

    except Exception as e:
        print(f"[Error] Unexcpected error while starting TensorBoard: {e}")


def str2bool(string):
    """
    Converts a string to a boolean value. Used in argument parsing.

    Args:
        string (str): String representation of a boolean value (e.g., 'true', 'false', '1', '0').

    Returns:
        bool: True if the string is 'true' (case-insentitive) or '1', False otherwise.
    """

    try:
        string = string.lower()
        if string in ["true", "1"]:
            return True
        elif string in ["false", "0"]:
            return False
    except ValueError as e:
        raise ValueError(f"[Error] Invalid boolean string `{string}` : {e}")


def str2list(string, separator="-", dtype=int):
    """
    Converts a separated string into a list of specified type. Used in argument parsing.

    Args:
        string (str): The input string to be converted. (e.g., '1-2-3').
        separator (str): The separator used in the string. Default is '-'.
        dtype (type): The type to which the elements in the list should be converted. Default is int.

    Returns:
        List[dtype]: A list of elements of the specified type.
    """

    try:
        return list(map(dtype, string.split(separator)))
    except ValueError as e:
        raise ValueError(f"[Error] Invalid list string `{string}` : {e}")
