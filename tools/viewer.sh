#!/usr/bin/env bash
# Opens the word viewer (every sign: its video next to the avatar) in the browser, starting what it needs first:
# the backend on :8010 (serves the gestures and the videos) and the Vite dev server on :5173 (or the next free port, if
# another project's server holds 5173: it answered /words.html with its own page).
#   tools/viewer.sh          start whatever is not running yet and open http://127.0.0.1:<port>/words.html
#   tools/viewer.sh servers  the same without opening the browser
#   tools/viewer.sh stop     stop the servers this script started
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
run="data/work/viewer"
mkdir -p "$run"

listening() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
# Whether the server on a port is this project's: its backend's API title, its Vite's words page.
ours_backend() { curl -s --max-time 2 "http://127.0.0.1:$1/openapi.json" | grep -q "VisualMind RSL Gesture API"; }
ours_frontend() { curl -s --max-time 2 "http://127.0.0.1:$1/words.html" | grep -q "/src/words.tsx"; }

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
    ours_backend 8010 || ours_frontend "$(cat "$run/frontend.port" 2>/dev/null || echo 5173)" || break
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

# The frontend asks for the backend at :8010 (frontend/src/App.tsx), so that port has to be ours.
if listening 8010 && ! ours_backend 8010; then
  echo "Порт 8010 занят чужим сервером, а фронтенд ищет бэкенд там: освободите его" >&2
  exit 1
fi
start backend 8010 env PYTHONPATH=backend .venv-linux/bin/python -m uvicorn rsl_gesture.api.app:app --host 127.0.0.1 --port 8010
# Ours where it already runs, else the first free port from 5173.
port=""
for candidate in $(seq 5173 5190); do
  if ours_frontend "$candidate" || ! listening "$candidate"; then
    port="$candidate"
    break
  fi
  echo "Порт $candidate занят другим сервером, беру следующий"
done
[[ -n "$port" ]] || { echo "Нет свободного порта 5173–5190" >&2; exit 1; }
start frontend "$port" bash -c "cd frontend && exec npx vite --host 127.0.0.1 --port $port --strictPort"
echo "$port" >"$run/frontend.port"

url="http://127.0.0.1:$port/words.html"
echo "$url"
[[ "${1:-}" == "servers" ]] || xdg-open "$url" >/dev/null 2>&1 || echo "Откройте адрес в браузере."
