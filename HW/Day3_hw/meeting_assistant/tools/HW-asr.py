#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
from pathlib import Path

import requests


def wait_download(url: str, auth: tuple[str, str], max_tries: int, poll_s: float) -> str | None:
    """Poll an endpoint until 200 OK, otherwise keep waiting (commonly 404 while processing)."""
    for _ in range(max_tries):
        try:
            resp = requests.get(url, timeout=(5, 60), auth=auth)
            if resp.status_code == 200:
                return resp.text
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(poll_s)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload WAV to ASR service and download TXT/SRT.")
    parser.add_argument("--audio", type=str, default="", help="Path to WAV file.")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for TXT/SRT.")
    parser.add_argument("--base", type=str, default="", help="ASR service base URL (e.g. https://3090api.huannago.com)")
    parser.add_argument("--user", type=str, default="", help="Basic auth username.")
    parser.add_argument("--password", type=str, default="", help="Basic auth password.")
    parser.add_argument("--max-tries", type=int, default=600, help="Max polling attempts.")
    parser.add_argument("--poll-s", type=float, default=2.0, help="Polling interval seconds.")
    parser.add_argument("--save-srt", action="store_true", help="Also save SRT if available.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent              # meeting_assistant/tools
    default_audio = (base_dir.parent / "input" / "Podcast_EP14.wav")
    audio_path = Path(args.audio).expanduser() if args.audio else default_audio
    audio_path = audio_path.resolve()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (base_dir / "out")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = (args.base or os.getenv("ASR_BASE") or "https://3090api.huannago.com").strip().rstrip("/")
    user = (args.user or os.getenv("ASR_USER") or "nutc2504").strip()
    pwd = (args.password or os.getenv("ASR_PASS") or "nutc2504").strip()
    auth = (user, pwd)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if audio_path.is_dir():
        raise IsADirectoryError(f"--audio points to a directory, not a file: {audio_path}")

    create_url = f"{base}/api/v1/subtitle/tasks"
    print(f"[HW-asr] base={base}")
    print(f"[HW-asr] audio={audio_path}")
    print(f"[HW-asr] out_dir={out_dir}")

    # 1) create task
    with open(audio_path, "rb") as f:
        r = requests.post(create_url, files={"audio": f}, timeout=60, auth=auth)
    r.raise_for_status()
    task_id = r.json()["id"]
    print("task_id:", task_id)
    print("等待轉文字...")

    txt_url = f"{base}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT"
    srt_url = f"{base}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    # 2) TXT
    txt_text = wait_download(txt_url, auth=auth, max_tries=args.max_tries, poll_s=args.poll_s)
    if txt_text is None:
        raise TimeoutError("TXT 轉錄逾時或服務端錯誤（一直拿不到 200）")

    txt_path = out_dir / f"{task_id}.txt"
    txt_path.write_text(txt_text, encoding="utf-8")
    print("轉錄成功:", txt_path)

    # 3) SRT (optional)
    if args.save_srt:
        srt_text = wait_download(srt_url, auth=auth, max_tries=args.max_tries, poll_s=args.poll_s)
        if srt_text is not None:
            srt_path = out_dir / f"{task_id}.srt"
            srt_path.write_text(srt_text, encoding="utf-8")
            print("轉錄成功:", srt_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
