import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
import os, time
import torch
from model import AIS_LSTM
from data import ShiftedMIMIC3SepsisDataModule

def main(hparams, cluster):
    # wait for Slurm file writes to propagate
    time.sleep(15)
    
    pl.seed_everything(0)
    
    dm = ShiftedMIMIC3SepsisDataModule()
    model = AIS_LSTM(
        dm.observation_dim,
        dm.context_dim,
        dm.num_actions, 
        latent_dim=hparams.latent_dim,
        lr=hparams.lr,
    )
    logger = CSVLogger("logs_shifted", name="AIS_LSTM_model_shifted")

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
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

def optimize_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='./slurm/',
    )

    # email for cluster coms
    cluster.notify_job_status(email='tangsp@umich.edu', on_done=False, on_fail=False)

    # configure cluster
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.job_time = '12:00:00'
    cluster.memory_mb_per_node = 0

    cluster.add_slurm_cmd(cmd='partition', value='gpu', comment='')
    cluster.add_slurm_cmd(cmd='mem', value='8GB', comment='')
    cluster.add_slurm_cmd(cmd='cpus-per-task', value='2', comment='')

    # run hopt
    # creates and submits jobs to slurm
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=25,
        job_name='mimic_sepsis_AIS_LSTM_shifted_job',
    )

if __name__ ==  '__main__':
    # Hyperparameter search grid
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--latent_dim', default=64, type=int, tunable=True, options=[8, 16, 32, 64, 128])
    parser.opt_list('--lr', default=1e-4, type=float, tunable=True, options=[1e-5, 5e-4, 1e-4, 5e-3, 1e-3])
    hparams = parser.parse_args()
    
    # Local execution: single trial (Tang used SLURM grid search)
    main(hparams, cluster=None)
