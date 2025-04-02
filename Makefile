.PHONY: install lint

install:
		git lfs install
		git clone https://www.modelscope.cn/black-forest-labs/FLUX.1-schnell.git pretrained_models/FLUX.1-schnell

lint:
		uv run ruff format
		uv run ruff check --fix