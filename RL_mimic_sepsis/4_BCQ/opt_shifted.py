import argparse
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data_shifted import EpisodicBuffer, SASRBuffer, add_data_specific_args, remap_rewards
from model import BCQ


class StopAndSave(pl.Callback):
    def __init__(self, n=100):
        self.n = n
        self._step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._step += 1
        if self._step % self.n == 0:
            path = os.path.join(trainer.log_dir, f"step={self._step}.ckpt")
            trainer.save_checkpoint(path)
        if self._step >= trainer.max_steps:
            trainer.should_stop = True


# Problem-specific hyperparameters
state_dim = 64
num_actions = 25
horizon = 20


def main(args):
    pl.seed_everything(args.seed)
    logger = CSVLogger("logs_shifted", name="mimic_dBCQ_shifted")

    train_buffer = SASRBuffer(state_dim, num_actions)
    train_buffer.load("../data/episodes+encoded_state+knn_pibs/shifted_train_data.pt")
    val_episodes = EpisodicBuffer(state_dim, num_actions, horizon)
    val_episodes.load("../data/episodes+encoded_state+knn_pibs/shifted_val_data.pt")

    train_buffer.reward = remap_rewards(train_buffer.reward, args)
    val_episodes.reward = remap_rewards(val_episodes.reward, args)
    Rmin = float(np.min(train_buffer.reward.cpu().numpy()))
    Rmax = float(np.max(train_buffer.reward.cpu().numpy()))

    train_buffer_loader = DataLoader(train_buffer, batch_size=100, shuffle=True)
    val_episodes_loader = DataLoader(
        val_episodes, batch_size=len(val_episodes), shuffle=False
    )

    policy = BCQ(
        state_dim=state_dim,
        num_actions=num_actions,
        Rmin=Rmin,
        Rmax=Rmax,
        **vars(args),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_steps=args.max_steps,
        logger=logger,
        val_check_interval=args.eval_frequency,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=[StopAndSave(n=args.eval_frequency)],
    )

    trainer.fit(policy, train_buffer_loader, val_episodes_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_data_specific_args(parser)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--eval_discount", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--target_value_clipping", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else [0, 1, 2, 3, 4]
    thresholds = (
        [args.threshold]
        if args.threshold is not None
        else [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.9999]
    )

    for threshold in thresholds:
        for seed in seeds:
            args.seed = seed
            args.threshold = threshold
            print(f"\n=== Starting: seed={seed}, threshold={threshold} ===")
            main(args)
