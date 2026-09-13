import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_simscale"


class MergedCacheDataset(Dataset):
    """Merge cache datasets while keeping the last sample for duplicate tokens."""

    def __init__(self, datasets: Sequence[CacheOnlyDataset]):
        if not datasets:
            raise ValueError("At least one cache dataset is required")

        samples: Dict[str, Tuple[CacheOnlyDataset, int]] = {}
        for dataset in datasets:
            for index, token in enumerate(dataset.tokens):
                samples[token] = (dataset, index)
        self._samples = list(samples.values())

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        dataset, dataset_index = self._samples[index]
        return dataset[dataset_index]


def _load_log_names(manifest_path: Path, split: str) -> List[str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SimScale log manifest does not exist: {manifest_path}")

    manifest = OmegaConf.load(manifest_path)
    key = f"{split}_logs"
    if key not in manifest:
        raise KeyError(f"Missing '{key}' in SimScale log manifest: {manifest_path}")
    return list(manifest[key])


def _build_cache_dataset(
    cache_path: Path,
    log_names: Sequence[str],
    agent: AbstractAgent,
) -> CacheOnlyDataset:
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Training cache does not exist: {cache_path}")
    return CacheOnlyDataset(
        cache_path=str(cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=list(log_names),
    )


def build_simscale_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """Build NAVSIM datasets augmented with one or more SimScale caches."""

    cache_paths = [Path(path) for path in cfg.simscale.cache_paths]
    manifest_paths = [Path(path) for path in cfg.simscale.log_manifest_paths]
    if len(cache_paths) != len(manifest_paths):
        raise ValueError("simscale.cache_paths and simscale.log_manifest_paths must have the same length")
    if not cache_paths:
        raise ValueError("At least one SimScale cache and log manifest must be configured")

    train_datasets = [
        _build_cache_dataset(Path(cfg.cache_path), cfg.train_logs, agent),
    ]
    val_datasets = [
        _build_cache_dataset(Path(cfg.cache_path), cfg.val_logs, agent),
    ]

    for cache_path, manifest_path in zip(cache_paths, manifest_paths):
        train_dataset = _build_cache_dataset(cache_path, _load_log_names(manifest_path, "train"), agent)
        val_dataset = _build_cache_dataset(cache_path, _load_log_names(manifest_path, "val"), agent)
        logger.info(
            "Loaded SimScale cache %s: %d training and %d validation samples",
            cache_path,
            len(train_dataset),
            len(val_dataset),
        )
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    return MergedCacheDataset(train_datasets), MergedCacheDataset(val_datasets)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Train an agent with NAVSIM and SimScale cached samples."""

    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global seed set to %d", cfg.seed)
    logger.info("Path where all results are stored: %s", cfg.output_dir)

    if not cfg.use_cache_without_dataset:
        raise ValueError("SimScale training requires use_cache_without_dataset=True")
    if cfg.force_cache_computation:
        raise ValueError("SimScale training requires force_cache_computation=False")

    logger.info("Building agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    lightning_module = AgentLightningModule(agent=agent)

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
