.PHONY: install lint

REGISTRY=fsl
IMAGE=flux-api
VERSION=latest

install:
		git lfs install
		git clone https://www.modelscope.cn/black-forest-labs/FLUX.1-dev.git pretrained_models/FLUX.1-dev

lint:
		uv run ruff format
		uv run ruff check --fix

build:
		docker build -t $(REGISTRY)/$(IMAGE):$(VERSION) .

run:
		docker run \
		-d \
		-p 50001:50001 \
		-v ./pretrained_models:/app/pretrained_models \
		--restart always \
		--name flux-api-1 \
		$(REGISTRY)/$(IMAGE):$(VERSION)
