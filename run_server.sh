#!/bin/bash
# Helper: jalankan main.py pakai pyenv Python (pasti ada torch+transformers).
# Usage: ./run_server.sh   atau   bash run_server.sh

cd "$(dirname "$0")"
exec /Users/miftahfaridlal-anshari/.pyenv/versions/3.10.18/bin/python main.py
