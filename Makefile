.PHONY: install lint

install:
		git lfs install
		git clone https://www.modelscope.cn/black-forest-labs/FLUX.1-dev.git pretrained_models/FLUX.1-dev
		git clone https://www.modelscope.cn/black-forest-labs/FLUX.1-Canny-dev-lora.git pretrained_models/FLUX.1-Canny-dev-lora

lint:
		uv run ruff format
		uv run ruff check --fix
