**Linode Deploy Steps (caesar_ocr API)**

1) Create Linode
- Ubuntu 24.04 LTS
- 2–4 GB RAM
- Add SSH key

2) DNS
- Point `api.yourdomain.com` -> Linode IP (A record)

3) Base setup
```
sudo apt update
sudo apt install -y docker.io caddy ufw
sudo usermod -aG docker $USER
newgrp docker
```

4) Deploy image
Option A (build on server):
```
git clone <your-repo>
cd caesar_ocr
docker build -t caesar-ocr:latest .
```

5) Environment file
Create `/etc/caesar-ocr.env`:
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-1
CAESAR_WARM_TOKEN_MODELS=false
```

6) systemd service
Create `/etc/systemd/system/caesar-ocr.service`:
```
[Unit]
Description=caesar_ocr API (Docker)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Restart=always
EnvironmentFile=/etc/caesar-ocr.env
ExecStartPre=-/usr/bin/docker rm -f caesar-ocr
ExecStart=/usr/bin/docker run --name caesar-ocr \
  -p 127.0.0.1:8000:8000 \
  --env-file /etc/caesar-ocr.env \
  caesar-ocr:latest
ExecStop=/usr/bin/docker stop caesar-ocr

[Install]
WantedBy=multi-user.target
```

Then:
```
sudo systemctl daemon-reload
sudo systemctl enable --now caesar-ocr
```

7) HTTPS via Caddy
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

8) Firewall
```
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```
