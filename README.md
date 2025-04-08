# flux-api

Providing an api for accessing self-hosting Flux model.

## Download model

```bash
git lfs install
git clone https://www.modelscope.cn/black-forest-labs/FLUX.1-dev.git pretrained_models/FLUX.1-dev
git clone https://www.modelscope.cn/black-forest-labs/FLUX.1-Canny-dev-lora.git pretrained_models/FLUX.1-Canny-dev-lora
```

## Run as system service

Copy the following system service configuration as `flux-api.service` into `/etc/systemd/system`

```bash
[Unit]
Description=FLUX API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=YOUR_WORKING_DIRECTORY
ExecStart=YOUR_WORKING_DIRECTORY/.venv/bin/python api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Start the system service

```bash
sudo systemctl daemon-reload # Reload systemd to apply changes
sudo systemctl start flux-api # Start the service
sudo systemctl enable flux-api # Optional: enable it to run on boot
```
