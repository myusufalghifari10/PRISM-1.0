"""
Retrain AIS_LSTM encoder with lr=5e-4 (matching Tang's best encoder).
Other params: latent_dim=64, seed=0, SWA, early stopping.
"""
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
from model import AIS_LSTM
from data import MIMIC3SepsisDataModule

def main():
    pl.seed_everything(0)
    
    dm = MIMIC3SepsisDataModule()
    model = AIS_LSTM(
        dm.observation_dim,
        dm.context_dim,
        dm.num_actions,
        latent_dim=64,
        lr=5e-4,  # Tang's best encoder lr
    )
    logger = CSVLogger("logs_orig_lr5e4", name="AIS_LSTM_model")

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=1000,
        callbacks=[
            pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=True),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss"),
            pl.callbacks.ModelCheckpoint(monitor="val_mse"),
        ],
    )
    trainer.fit(model, dm)
    
    # Report results
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Best checkpoint (val_loss): {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score}")
    print(f"Epochs trained: {trainer.current_epoch}")

if __name__ == '__main__':
    main()
