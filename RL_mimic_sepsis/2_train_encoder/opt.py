import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import os
from model import AIS_LSTM
from data import MIMIC3SepsisDataModule

def main(hparams):
    pl.seed_everything(0)
    
    dm = MIMIC3SepsisDataModule()
    model = AIS_LSTM(
        dm.observation_dim,
        dm.context_dim,
        dm.num_actions, 
        latent_dim=hparams.latent_dim,
        lr=hparams.lr,
    )
    logger = CSVLogger("logs", name="AIS_LSTM_model")

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu' if __import__('torch').cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=1000,
        callbacks=[
            pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=False),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss"),
            pl.callbacks.ModelCheckpoint(monitor="val_mse"),
        ],
    )
    trainer.fit(model, dm)

if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    hparams = parser.parse_args()
    
    main(hparams)
