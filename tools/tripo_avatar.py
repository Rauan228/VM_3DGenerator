"""Make a rigged avatar with the Tripo API (v3), one step at a time, so each paid step can be checked before the next.

    python tools/tripo_avatar.py restyle --image ref.png --prompt-file prompt.txt --out models/tripo/male   # ~10 credits
    python tools/tripo_avatar.py model --input <image task id | image path> --out models/tripo/male          # ~40 credits
    python tools/tripo_avatar.py rig --input <model task id> --out models/tripo/male                         # 25 credits
    python tools/tripo_avatar.py balance

The API key is read from TRIPO_API_KEY or ~/.config/visualmind/tripo.env and is never printed. Every task's JSON is
saved next to the output (<out>_<step>_task.json) together with what it produced.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

API = "https://openapi.tripo3d.ai/v3"
KEY_FILE = os.path.expanduser("~/.config/visualmind/tripo.env")


def api_key() -> str:
    key = os.environ.get("TRIPO_API_KEY")
    if not key and os.path.exists(KEY_FILE):
        for line in open(KEY_FILE):
            if line.startswith("TRIPO_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit(f"TRIPO_API_KEY is not set (env or {KEY_FILE})")
    return key


def request(method: str, path: str, body: bytes | None = None, content_type: str = "application/json") -> dict:
    req = urllib.request.Request(f"{API}{path}", data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key()}")
    if body is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"{method} {path}: HTTP {error.code} {error.read().decode(errors='replace')[:600]}")
    if data.get("code") != 0:
        sys.exit(f"{method} {path}: {data}")
    return data["data"]


def upload(path: str) -> str:
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    return request("POST", "/files", body, f"multipart/form-data; boundary={boundary}")["file_token"]


def run(endpoint: str, payload: dict, label: str, out: str) -> dict:
    task_id = request("POST", endpoint, json.dumps(payload).encode())["task_id"]
    print(f"{label}: {task_id}", flush=True)
    last = None
    while True:
        task = request("GET", f"/tasks/{task_id}")
        status = task.get("status")
        if status != last:
            print(f"  {status}", flush=True)
            last = status
        if status == "success":
            break
        if status in ("failed", "cancelled", "banned", "expired"):
            sys.exit(f"{label} {status}: {task.get('error_code')} {task.get('error_message')}")
        time.sleep(3)
    json.dump(task, open(f"{out}_{label}_task.json", "w"), indent=1)
    for key, url in (task.get("output") or {}).items():
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
        path = f"{out}_{label}_{key}{ext}"
        urllib.request.urlretrieve(url, path)
        print(f"  saved {path}")
    print(f"  credits: {task.get('credits_consumed')} · balance {request('GET', '/account/balance')['balance']}")
    return task


def as_input(value: str) -> str:
    return upload(value) if os.path.exists(value) else value


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    restyle = sub.add_parser("restyle")
    restyle.add_argument("--image", required=True)
    restyle.add_argument("--prompt-file", required=True)
    restyle.add_argument("--out", required=True)
    restyle.add_argument("--model", default="chat_image_2")
    model = sub.add_parser("model")
    model.add_argument("--input", required=True, help="image path, file token or image task id")
    model.add_argument("--out", required=True)
    model.add_argument("--texture-quality", default="detailed")
    rig = sub.add_parser("rig")
    rig.add_argument("--input", required=True, help="model task id")
    rig.add_argument("--out", required=True)
    sub.add_parser("balance")
    args = parser.parse_args()

    if args.command == "balance":
        print(request("GET", "/account/balance"))
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.command == "restyle":
        run("/generation/image-to-image", {
            "model": args.model,
            "input": upload(args.image),
            "prompt": open(args.prompt_file).read().strip(),
            "size": "1024x1536",
            "quality": "medium",
        }, "restyle", args.out)
    elif args.command == "model":
        run("/generation/image-to-model", {
            "input": as_input(args.input),
            "model": "v3.1-20260211",
            "texture": True,
            "pbr": True,
            "texture_quality": args.texture_quality,
            "face_limit": 100000,
        }, "model", args.out)
    else:
        check = run("/animations/rig-check", {"input": args.input}, "rigcheck", args.out)
        output = check.get("output") or {}
        print("  rig check:", json.dumps(output))
        if output.get("riggable") is False:
            sys.exit("Tripo says the model is not riggable")
        run("/animations/rig", {
            "input": args.input,
            "model": "v3.0-20260909",
            "rig_type": output.get("rig_type", "biped"),
            "spec": "mixamo",
            "out_format": "glb",
        }, "rig", args.out)


if __name__ == "__main__":
    main()
