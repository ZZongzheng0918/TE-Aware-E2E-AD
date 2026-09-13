import pytorch_lightning as pl

from navsim.agents.diffusiondrive_simscale.transfuser_te_config import TransfuserTEConfig


class TransfuserTECallback(pl.Callback):
    """No-op callback retained for the DiffusionDrive agent interface."""

    def __init__(self, config: TransfuserTEConfig) -> None:
        self._config = config
