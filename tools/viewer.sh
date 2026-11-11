#!/usr/bin/env bash
# Opens the word viewer (every sign: its video next to the avatar) in the browser, starting what it needs first:
# the backend on :8010 (serves the gestures and the videos) and the Vite dev server on :5173.
#   tools/viewer.sh          start whatever is not running yet and open http://127.0.0.1:5173/words.html
#   tools/viewer.sh servers  the same without opening the browser
#   tools/viewer.sh stop     stop the servers this script started
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
run="data/work/viewer"
url="http://127.0.0.1:5173/words.html"
mkdir -p "$run"

listening() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

if [[ "${1:-}" == "stop" ]]; then
  for name in backend frontend; do
    if [[ -f "$run/$name.pid" ]]; then
      # The server is the leader of its own session: the whole group goes (Vite runs under npx).
      kill -- "-$(cat "$run/$name.pid")" 2>/dev/null && echo "Остановлен: $name" || true
      rm -f "$run/$name.pid"
    fi
  done
  # A server takes a moment to let its port go; started again before that, it would be taken for still running.
  for _ in $(seq 20); do
    listening 8010 || listening 5173 || break
    sleep 0.25
  done
  exit 0
fi

# start <name> <port> <command...>: nothing to do when the port already answers.
start() {
  local name="$1" port="$2"
  shift 2
  if listening "$port"; then
    echo "$name: уже работает на :$port"
    return
  fi
  setsid "$@" >"$run/$name.log" 2>&1 </dev/null &
  echo $! >"$run/$name.pid"
  for _ in $(seq 60); do
    listening "$port" && { echo "$name: запущен на :$port"; return; }
    sleep 0.5
  done
  echo "$name: не запустился за 30 с, см. $root/$run/$name.log" >&2
  exit 1
}

[[ -x .venv-linux/bin/python ]] || { echo "Нет .venv-linux (окружение бэкенда)" >&2; exit 1; }
[[ -d frontend/node_modules ]] || { echo "Нет frontend/node_modules: выполните npm ci в frontend/" >&2; exit 1; }

start backend 8010 env PYTHONPATH=backend .venv-linux/bin/python -m uvicorn rsl_gesture.api.app:app --host 127.0.0.1 --port 8010
start frontend 5173 bash -c "cd frontend && exec npx vite --host 127.0.0.1 --port 5173"

echo "$url"
[[ "${1:-}" == "servers" ]] || xdg-open "$url" >/dev/null 2>&1 || echo "Откройте адрес в браузере."
