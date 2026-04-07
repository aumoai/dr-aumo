from __future__ import annotations

import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"
try:
    import deepspeed  # noqa: F401
except Exception:
    pass

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.model_utils import ModelConfig
from open_instruct.utils import ArgumentParserPlus

from .config import GRPOConfig
from .trainer import GRPOTrainer


def main() -> None:
    parser = ArgumentParserPlus((GRPOConfig, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    trainer = GRPOTrainer(args, tokenizer_config, model_config)
    trainer.train()


if __name__ == "__main__":
    main()
