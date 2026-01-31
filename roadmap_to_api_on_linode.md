**Roadmap: caesar_ocr API on Linode (Docker + systemd + Caddy)**

1) VM setup (Ubuntu 22.04/24.04)
```
sudo apt update
sudo apt install -y docker.io caddy
sudo usermod -aG docker $USER
newgrp docker
```

2) Build or pull the image
Option A (build on server):
```
git clone <your-repo>
cd caesar_ocr
docker build -t caesar-ocr:latest .
```

Option B (pull from registry):
```
docker pull <your-registry>/caesar-ocr:latest
docker tag <your-registry>/caesar-ocr:latest caesar-ocr:latest
```

3) Create a systemd service
Create `/etc/systemd/system/caesar-ocr.service`:
```
[Unit]
Description=caesar_ocr API (Docker)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f caesar-ocr
ExecStart=/usr/bin/docker run --name caesar-ocr \
  -p 127.0.0.1:8000:8000 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=... \
  -e CAESAR_WARM_TOKEN_MODELS=true \
  caesar-ocr:latest
ExecStop=/usr/bin/docker stop caesar-ocr

[Install]
WantedBy=multi-user.target
```

Then:
```
sudo systemctl daemon-reload
sudo systemctl enable --now caesar-ocr
sudo systemctl status caesar-ocr --no-pager
```

4) Caddy reverse proxy (HTTPS)
Edit `/etc/caddy/Caddyfile`:
```
api.yourdomain.com {
  reverse_proxy 127.0.0.1:8000
}
```

Then:
```
sudo systemctl reload caddy
```

5) Firewall
```
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

Notes:
- Store AWS credentials as environment variables or use an EnvironmentFile in systemd.
- For updates: `docker pull ...`, then `sudo systemctl restart caesar-ocr`.
