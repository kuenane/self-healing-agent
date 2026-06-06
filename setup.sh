#!/usr/bin/env bash
# setup.sh — Bootstrap the Arize Self-Healing Agent
# Usage: bash setup.sh
set -e

echo "=== Arize Self-Healing Agent setup ==="

# 1. Python virtualenv
if [ ! -d ".venv" ]; then
  echo "[1/5] Creating virtualenv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install Python deps
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Phoenix MCP server (Node.js required)
echo "[3/5] Checking Node.js / npx..."
if command -v npx &>/dev/null; then
  echo "  npx found — Phoenix MCP server ready."
  echo "  Start it with:  npx @arizeai/phoenix-mcp --port 6006"
else
  echo "  WARNING: Node.js not found. Install from https://nodejs.org"
fi

# 4. .env file
if [ ! -f ".env" ]; then
  echo "[4/5] Creating .env from template..."
  cp .env.example .env
  echo "  ⚠  Edit .env and add your PHOENIX_API_KEY and GEMINI_API_KEY"
else
  echo "[4/5] .env already exists — skipping."
fi

# 5. Redis check
echo "[5/5] Checking Redis..."
if command -v redis-cli &>/dev/null && redis-cli ping &>/dev/null 2>&1; then
  echo "  Redis is running."
else
  echo "  Redis not running — agent will use in-memory state."
  echo "  Install Redis: sudo apt install redis-server  (or brew install redis)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Start Phoenix:  npx @arizeai/phoenix-mcp --port 6006"
echo "  3. Run the agent:  python main.py"
