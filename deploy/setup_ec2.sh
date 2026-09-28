#!/usr/bin/env bash
# Provision (or re-provision, after a git pull) Astra on a fresh Ubuntu 24.04 EC2
# instance. Safe to re-run: re-running after a `git pull` is how updates get
# deployed. Run this ON the EC2 instance, over SSH -- not on your own machine.
#
# Usage:
#   ssh -i <your-key>.pem ubuntu@<ec2-public-ip>
#   git clone https://github.com/Imro-iitr6394/Call-Intelligence-AI-Agent.git astra
#   cd astra
#   bash deploy/setup_ec2.sh

set -euo pipefail

REPO_DIR="/home/ubuntu/astra"
SERVICE_NAME="astra"

echo "==> Installing system packages (python3.12-venv, git)"
sudo apt update -y
sudo apt install -y python3.12-venv git

cd "$REPO_DIR"

echo "==> Creating/updating the virtualenv"
if [ ! -d ".venv" ]; then
  python3.12 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f ".env" ]; then
  echo "==> No .env found -- copying the template. You MUST edit this before the app will do anything useful:"
  cp deploy/.env.example .env
  chmod 600 .env
  echo "    nano .env"
fi

echo "==> Installing the systemd service"
sudo cp "deploy/${SERVICE_NAME}.service" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "==> Done. Check status with: sudo systemctl status ${SERVICE_NAME}"
echo "==> Tail logs with:          sudo journalctl -u ${SERVICE_NAME} -f"
