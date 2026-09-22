#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$DIR/backend"
FRONTEND_DIR="$DIR/frontend"
DB_FILE="$BACKEND_DIR/database.db"

# Cleanup trap for graceful shutdown
cleanup() {
    echo -e "\n[!] Shutting down TrueTone Demo..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "    TrueTone Voice-Clone Detection"
echo "=========================================="

if [[ "$1" == "--reset" ]]; then
    echo "[*] Resetting database and state..."
    rm -f "$DB_FILE"
    echo "[+] Database reset complete."
fi

echo "[*] Starting Backend (FastAPI)..."
cd "$BACKEND_DIR"
PYTHONPATH="$BACKEND_DIR" ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[*] Starting Frontend (Next.js) on PORT 5000..."
cd "$FRONTEND_DIR"
PORT=5000 npm run dev &
FRONTEND_PID=$!

echo "=========================================="
echo "[+] Services starting!"
echo "    Frontend / Dashboard : http://localhost:5000"
echo "    Backend API          : http://localhost:8000"
echo "    Press Ctrl+C to stop both services."
echo "=========================================="

# Wait for background jobs
wait $BACKEND_PID
wait $FRONTEND_PID
