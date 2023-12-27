from pytorch_lightning.callbacks import ModelCheckpoint


class SaveLastCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_last_checkpoint(trainer, monitor_candidates)
