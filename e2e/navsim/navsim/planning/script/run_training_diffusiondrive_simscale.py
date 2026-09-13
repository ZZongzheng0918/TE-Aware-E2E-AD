import logging

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.script.run_training_simscale import build_simscale_datasets
from navsim.planning.training.custom_agent_lightning_module import CustomAgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_simscale"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a DiffusionDrive agent with NAVSIM and SimScale cached samples."""

    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global seed set to %d", cfg.seed)
    logger.info("Path where all results are stored: %s", cfg.output_dir)

    if not cfg.use_cache_without_dataset:
        raise ValueError("SimScale training requires use_cache_without_dataset=True")
    if cfg.force_cache_computation:
        raise ValueError("SimScale training requires force_cache_computation=False")

    logger.info("Building DiffusionDrive agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    lightning_module = CustomAgentLightningModule(agent=agent)

    logger.info("Building NAVSIM and SimScale datasets")
    train_data, val_data = build_simscale_datasets(cfg, agent)
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    callbacks = [
        callback for callback in agent.get_training_callbacks() if not isinstance(callback, ModelCheckpoint)
    ]
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename="{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
        )
    )

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
