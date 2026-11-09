"""Downloads the trimmed Slovo recordings of the given words (all 20 per word) out of the remote zip, 8 at a time.

    python3 tools/slovo/fetch.py words.txt     # one word per line, as Slovo writes it

Needs data/slovo/annotations.csv and names.txt (`remote_zip.py get URL annotations.csv`, `remote_zip.py list URL`).
Writes data/slovo/videos/<attachment_id>.mp4.
"""
import csv, sys, threading, queue
from pathlib import Path
from remote_zip import open_zip

URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "slovo"
OUT = DATA / "videos"; OUT.mkdir(exist_ok=True)
words = {w.strip() for w in open(sys.argv[1]) if w.strip()}
rows = [r for r in csv.DictReader(open(DATA / "annotations.csv"), delimiter="\t") if r["text"] in words]
names = {l.split()[0].split("/")[-1][:-4]: l.split()[0] for l in open(DATA / "names.txt") if ".mp4" in l}
todo = queue.Queue()
for r in rows:
    if not (OUT / f"{r['attachment_id']}.mp4").exists(): todo.put(r["attachment_id"])
print("to fetch", todo.qsize(), flush=True)
done = [0]; lock = threading.Lock()

def worker():
    z = open_zip(URL)
    while True:
        try: aid = todo.get_nowait()
        except queue.Empty: return
        try:
            tmp = OUT / f"{aid}.part"
            with z.open(names[aid]) as src, open(tmp, "wb") as dst: dst.write(src.read())
            tmp.rename(OUT / f"{aid}.mp4")
        except Exception as e:
            print("FAIL", aid, e, flush=True)
        with lock:
            done[0] += 1
            if done[0] % 200 == 0: print("done", done[0], flush=True)

ts = [threading.Thread(target=worker) for _ in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
print("finished", len(list(OUT.glob("*.mp4"))), flush=True)
