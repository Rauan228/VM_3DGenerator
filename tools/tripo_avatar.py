"""Generate a rigged avatar with the Tripo API from four views (front, left, back, right).

    python tools/tripo_avatar.py generate --views front.png left.png back.png right.png --out models/tripo/male
    python tools/tripo_avatar.py rig --task <model task id> --out models/tripo/male
    python tools/tripo_avatar.py balance

The API key is read from TRIPO_API_KEY, or from ~/.config/visualmind/tripo.env; it is never printed. "left" is the
view from the character's left side (the character faces the image's left), "right" the opposite.
Each stage saves its task JSON next to the output, so a later stage (or a rerun) does not pay for the earlier one again.
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

API = "https://api.tripo3d.ai/v2/openapi"
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
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"{method} {path}: HTTP {error.code} {error.read().decode(errors='replace')[:500]}")
    if data.get("code") != 0:
        sys.exit(f"{method} {path}: {data}")
    return data["data"]


def upload(path: str) -> str:
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path)[0] or "image/png"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    return request("POST", "/upload", body, f"multipart/form-data; boundary={boundary}")["image_token"]


def run_task(payload: dict, label: str) -> dict:
    task_id = request("POST", "/task", json.dumps(payload).encode())["task_id"]
    print(f"{label}: task {task_id}", flush=True)
    while True:
        task = request("GET", f"/task/{task_id}")
        status = task.get("status")
        print(f"  {status} {task.get('progress', '')}%", flush=True)
        if status == "success":
            return task
        if status in ("failed", "banned", "expired", "cancelled", "unknown"):
            sys.exit(f"{label} {status}: {json.dumps(task)[:800]}")
        time.sleep(8)


def download(task: dict, out: str):
    output = task.get("output", {})
    saved = []
    for key in ("pbr_model", "model", "base_model", "rendered_image"):
        url = output.get(key)
        if not url:
            continue
        ext = ".webp" if key == "rendered_image" else os.path.splitext(url.split("?")[0])[1] or ".glb"
        path = f"{out}_{key}{ext}"
        urllib.request.urlretrieve(url, path)
        saved.append(path)
        print(f"  saved {path} ({os.path.getsize(path)} bytes)")
    return saved


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--views", nargs=4, required=True, metavar=("FRONT", "LEFT", "BACK", "RIGHT"))
    gen.add_argument("--out", required=True)
    gen.add_argument("--model-version", default="v2.5-20250123")
    rig = sub.add_parser("rig")
    rig.add_argument("--task", required=True)
    rig.add_argument("--out", required=True)
    sub.add_parser("balance")
    args = parser.parse_args()

    if args.command == "balance":
        print(request("GET", "/user/balance"))
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.command == "generate":
        files = [{"type": "png", "file_token": upload(path)} for path in args.views]
        task = run_task({
            "type": "multiview_to_model",
            "files": files,
            "model_version": args.model_version,
            "texture": True,
            "pbr": True,
            "texture_quality": "standard",
            "orientation": "default",
        }, "model")
        json.dump(task, open(f"{args.out}_model_task.json", "w"), indent=1)
        download(task, f"{args.out}_model")
    else:
        check = run_task({"type": "animate_prerigcheck", "original_model_task_id": args.task}, "rig check")
        print("  rig check:", json.dumps(check.get("output", {})))
        if not check.get("output", {}).get("riggable", True):
            sys.exit("Tripo says the model is not riggable")
        task = run_task({
            "type": "animate_rig",
            "original_model_task_id": args.task,
            "out_format": "glb",
            "rig_type": check.get("output", {}).get("rig_type", "biped"),
            "spec": "mixamo",
        }, "rig")
        json.dump(task, open(f"{args.out}_rig_task.json", "w"), indent=1)
        download(task, f"{args.out}_rig")
    print(request("GET", "/user/balance"))


if __name__ == "__main__":
    main()
