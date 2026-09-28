# Deploying Astra to AWS EC2

One-time setup and day-to-day operation for running Astra on a single EC2 instance.
See the repo root [README.md](../README.md) for what the app does; this file is
deploy-only.

## First-time setup

1. Launch an EC2 instance:
   - AMI: **Ubuntu 24.04 LTS** (ships Python 3.12).
   - Instance type: **t3.small** or larger (the embedding model used for search needs
     real memory alongside Streamlit -- a 1&nbsp;GiB `t3.micro` risks OOM).
   - Storage: 20&nbsp;GiB gp3 root volume.
   - Security group: allow **SSH (22)** from your IP only, and **TCP 8501** from
     `0.0.0.0/0` (so the app is reachable at `http://<public-ip>:8501`).

2. SSH in and provision:

   ```bash
   ssh -i <your-key>.pem ubuntu@<ec2-public-ip>
   git clone https://github.com/Imro-iitr6394/Call-Intelligence-AI-Agent.git astra
   cd astra
   bash deploy/setup_ec2.sh
   ```

   The script installs system packages, creates a venv, installs
   `requirements.txt`, copies `.env.example` to `.env` (you'll need to edit it --
   see below), and installs + starts the `astra` systemd service.

3. Fill in real secrets:

   ```bash
   nano .env
   ```

   Set `TRANSCRIPTION_API_KEY` (AssemblyAI) and `GEMINI_API_KEY` (Gemini). Both
   `_2`/`_3` Gemini keys are optional quota-rotation backups. Never paste real
   values anywhere but this file -- it's `chmod 600` and gitignored.

   ```bash
   sudo systemctl restart astra
   ```

4. Verify:

   ```bash
   sudo systemctl status astra        # should read "active (running)"
   sudo journalctl -u astra -f        # first start is slower: the search
                                       # model downloads to ~/.cache/huggingface
   ```

   Then open `http://<ec2-public-ip>:8501` from your own browser.

## Redeploying an update

```bash
ssh -i <your-key>.pem ubuntu@<ec2-public-ip>
cd astra
git pull
bash deploy/setup_ec2.sh   # safe to re-run; only reinstalls what changed
```

## Operating

| Task | Command |
|---|---|
| Check status | `sudo systemctl status astra` |
| Tail logs | `sudo journalctl -u astra -f` |
| Restart | `sudo systemctl restart astra` |
| Stop | `sudo systemctl stop astra` |

## Data persistence

`data/call_intelligence.sqlite3` and `data/sources/` live on the instance's root EBS
volume. They survive reboots but are lost if the instance is **terminated** (not just
stopped). For anything beyond a throwaway demo, take a periodic EBS snapshot
(AWS Console -> Volumes -> Actions -> Create snapshot) or `scp` the `data/` folder down.

## Known tradeoff

This setup serves plain HTTP with no authentication -- anyone with the instance's IP
and port 8501 can open the app. That's an acceptable tradeoff for a demo box, not for
handling real customer call data in production. Adding a domain + HTTPS (Nginx +
Let's Encrypt) and/or an auth layer is a natural next step, not covered here.
