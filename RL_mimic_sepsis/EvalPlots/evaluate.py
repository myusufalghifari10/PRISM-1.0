import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EpisodicBufferO(Dataset):
    def __init__(self, state_dim, num_actions, horizon, buffer_size=0):
        self.max_size = int(buffer_size)
        self.horizon = horizon
        self.state = torch.zeros((self.max_size, horizon, state_dim))
        self.action = torch.zeros((self.max_size, horizon, 1), dtype=torch.long)
        self.reward = torch.zeros((self.max_size, horizon, 1))
        self.not_done = torch.zeros((self.max_size, horizon, 1))
        self.pibs = torch.zeros((self.max_size, horizon, num_actions))
        self.estm_pibs = torch.zeros((self.max_size, horizon, num_actions))
    
    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.not_done[idx],
            self.pibs[idx],
            self.estm_pibs[idx],
        )
    
    def load(self, filename):
        data = torch.load(filename)
        self.state = data['statevecs'][:, :-1, :]
        self.action = data['actions'][:, 1:].unsqueeze(-1)  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.reward = data['rewards'][:, 1:].unsqueeze(-1)  # Need to offset by 1
        self.not_done = data['notdones'][:, 1:].unsqueeze(-1)
        self.pibs = data['pibs'][:, :-1, :]
        self.estm_pibs = data['estm_pibs'][:, :-1, :]
        print(f"Episodic Buffer loaded with {len(self)} episides.")

import torch.nn.functional as F
def offline_evaluation_O(self, eval_buffer, weighted=True, eps=0.01):
    states, actions, rewards, not_dones, pibs, estm_pibs = eval_buffer
    rewards = rewards[:, :, 0].cpu().numpy()
    n, horizon, _ = states.shape
    discounted_rewards = rewards * (self.eval_discount ** np.arange(horizon))

    ir = np.ones((n, horizon))
    for idx in range(n):
        lng = (not_dones[idx, :, 0].sum() + 1).item()  # all but the final transition has notdone==1

        # Predict Q-values and Imitation probabilities
        q, _, i = self.Q(states[idx])
        imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
        imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()

        # Factored action remapping
        q = q @ self.all_subactions_vec.T
        imt = torch.einsum('bi,bj->bji', (imt[:,0,:], imt[:,1,:])).reshape(-1, 25)

        # Use large negative number to mask actions from argmax
        a_id = (imt * q + (1. - imt) * torch.finfo().min).argmax(axis=1).cpu().numpy()
        pie_soft = np.zeros((horizon, 25))
        pie_soft += eps * estm_pibs[idx].cpu().numpy() # Soften using training behavior policy
        pie_soft[range(horizon), a_id] += (1.0 - eps)

        # Compute importance sampling ratios
        a_obs = actions[idx, :, 0]
        ir[idx, :lng] = pie_soft[range(lng), a_obs[:lng].cpu().numpy()] / pibs[idx, range(lng), a_obs[:lng]].cpu().numpy()
        ir[idx, lng:] = 1  # Mask out the padded timesteps

    weights = np.clip(ir.cumprod(axis=1), 0, 1e3)
    if weighted:
        weights_norm = weights.sum(axis=0)
    else:
        weights_norm = weights.shape[0]
    weights /= weights_norm

    ess = (weights[:,-1].sum()) ** 2 / (((weights[:,-1]) ** 2).sum())
    estm = (weights[:,-1] * discounted_rewards.sum(axis=-1)).sum()

    return estm, ess

class EpisodicBufferF(Dataset):
    """
    Factored episodic buffer for offline evaluation.  Returns 8-item tuples:
      (state, action, subaction, subactionvec, reward, not_done, subpibs, estm_subpibs)

    ⚠️  FIELD-ORDER MISMATCH WARNING:
    SASRBuffer (training) returns: (state, action, subaction, subactionvec, next_state, reward, notdone, subpibs)
    EpisodicBufferF returns:       (state, action, subaction, subactionvec, reward, not_done, subpibs, estm_subpibs)

    Position 4: SASRBuffer → next_state | EpisodicBufferF → reward
    Position 5: SASRBuffer → reward     | EpisodicBufferF → not_done
    Position 6: SASRBuffer → notdone    | EpisodicBufferF → subpibs
    Position 7: SASRBuffer → subpibs    | EpisodicBufferF → estm_subpibs

    This is harmless CURRENTLY because:
    - training_step uses SASRBuffer (correct unpacking)
    - validation_step uses EpisodicBufferF (correct via model.offline_evaluation)
    - model.offline_evaluation explicitly unpacks by name, not position

    If you ever feed EpisodicBufferF into training_step, the mismatched fields will cause
    silent garbage. Use the `episodic_to_sasr` adapter to convert if needed.
    """
    def __init__(self, state_dim, num_actions, horizon, buffer_size=0):
        self.max_size = int(buffer_size)
        self.horizon = horizon
        self.state = torch.zeros((self.max_size, horizon, state_dim))
        self.action = torch.zeros((self.max_size, horizon, 1), dtype=torch.long)
        self.subaction = torch.zeros((self.max_size, horizon, 2), dtype=torch.long)
        self.subactionvec = torch.zeros((self.max_size, horizon, 10))
        self.reward = torch.zeros((self.max_size, horizon, 1))
        self.not_done = torch.zeros((self.max_size, horizon, 1))
        self.subpibs = torch.zeros((self.max_size, horizon, 10))
        self.estm_subpibs = torch.zeros((self.max_size, horizon, 10))
    
    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.action[idx],
            self.subaction[idx],
            self.subactionvec[idx],
            self.reward[idx],
            self.not_done[idx],
            self.subpibs[idx],
            self.estm_subpibs[idx],
        )
    
    def load(self, filename):
        data = torch.load(filename)
        self.state = data['statevecs'][:, :-1, :]
        self.action = data['actions'][:, 1:].unsqueeze(-1)  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.subaction = data['subactions'][:, 1:, :]  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.subactionvec = data['subactionvecs'][:, 1:, :]  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.reward = data['rewards'][:, 1:].unsqueeze(-1)  # Need to offset by 1
        self.not_done = data['notdones'][:, 1:].unsqueeze(-1)
        self.subpibs = data['subpibs'][:, :-1, :]
        self.estm_subpibs = data['estm_subpibs'][:, :-1, :]
        print(f"Episodic Buffer loaded with {len(self)} episides.")

def episodic_to_sasr(batch):
    """
    Adapter: convert EpisodicBufferF batch → SASRBuffer-compatible batch.

    EpisodicBufferF returns:   (state, action, subaction, subactionvec, reward,   not_done,  subpibs, estm_subpibs)
    SASRBuffer/training expects: (state, action, subaction, subactionvec, next_state, reward, notdone, subpibs)

    Computes next_state from state[:, 1:, :] and discards estm_subpibs.
    Should ONLY be used if feeding EpisodicBufferF into training_step.
    """
    state, action, subaction, subactionvec, reward, not_done, subpibs, estm_subpibs = batch
    # Compute next_state: state at t+1 (same offset pattern as load method)
    # For episodic data, next_state is just state shifted by 1 timestep
    next_state = torch.zeros_like(state)
    next_state[:, :-1, :] = state[:, 1:, :]
    # Reorder to match SASRBuffer / training_step unpacking order
    return (state, action, subaction, subactionvec, next_state, reward, not_done, subpibs)


# ── Shifted episodic buffer (no +1 offset) ──────────────────────────────────
class EpisodicBufferFShifted(Dataset):
    """
    Factored episodic buffer for SHIFTED data (no +1 offset on load).
    Use with data from SplitSepsisCohort_shifted.ipynb.
    """
    def __init__(self, state_dim, num_actions, horizon, buffer_size=0):
        self.max_size = int(buffer_size)
        self.horizon = horizon
        self.state = torch.zeros((self.max_size, horizon, state_dim))
        self.action = torch.zeros((self.max_size, horizon, 1), dtype=torch.long)
        self.subaction = torch.zeros((self.max_size, horizon, 2), dtype=torch.long)
        self.subactionvec = torch.zeros((self.max_size, horizon, 10))
        self.reward = torch.zeros((self.max_size, horizon, 1))
        self.not_done = torch.zeros((self.max_size, horizon, 1))
        self.subpibs = torch.zeros((self.max_size, horizon, 10))
        self.estm_subpibs = torch.zeros((self.max_size, horizon, 10))

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.action[idx],
            self.subaction[idx],
            self.subactionvec[idx],
            self.reward[idx],
            self.not_done[idx],
            self.subpibs[idx],
            self.estm_subpibs[idx],
        )

    def load(self, filename):
        data = torch.load(filename)
        # NO +1 offset — shifted data already pre-shifted
        self.state = data['statevecs'][:, :-1, :]
        self.action = data['actions'][:, :-1].unsqueeze(-1)
        self.subaction = data['subactions'][:, :-1, :]
        self.subactionvec = data['subactionvecs'][:, :-1, :]
        self.reward = data['rewards'][:, :-1].unsqueeze(-1)
        self.not_done = data['notdones'][:, :-1].unsqueeze(-1)
        self.subpibs = data['subpibs'][:, :-1, :]
        self.estm_subpibs = data['estm_subpibs'][:, :-1, :]
        print(f"Shifted Episodic Buffer loaded with {len(self)} episodes.")


def offline_evaluation_F(self, eval_buffer, weighted=True, eps=0.01):
    states, actions, subactions, subactionvecs, rewards, not_dones, subpibs, estm_subpibs = eval_buffer
    rewards = rewards[:, :, 0].cpu().numpy()
    n, horizon, _ = states.shape
    discounted_rewards = rewards * (self.eval_discount ** np.arange(horizon))

    ir = np.ones((n, horizon))
    for idx in range(n):
        lng = (not_dones[idx, :, 0].sum() + 1).item()  # all but the final transition has notdone==1

        # Predict Q-values and Imitation probabilities
        q, _, i = self.Q(states[idx])
        imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
        imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()

        # Factored action remapping
        q = q @ self.all_subactions_vec.T
        imt = torch.einsum('bi,bj->bji', (imt[:,0,:], imt[:,1,:])).reshape(-1, 25)

        # Use large negative number to mask actions from argmax
        a_id = (imt * q + (1. - imt) * torch.finfo().min).argmax(axis=1).cpu().numpy()
        pie_soft = np.zeros((horizon, 25))
        estm_pibs = np.einsum('bi,bj->bji', estm_subpibs[idx][:,:5].cpu().numpy(), estm_subpibs[idx][:,5:].cpu().numpy()).reshape((-1, 25))
        pie_soft += eps * estm_pibs # Soften using training behavior policy
        pie_soft[range(horizon), a_id] += (1.0 - eps)

        # Compute importance sampling ratios
        a_obs = actions[idx, :, 0]
        ir[idx, :lng] = pie_soft[range(lng), a_obs[:lng].cpu().numpy()] / \
            (subpibs[idx, range(lng), a_obs[:lng] % 5].cpu().numpy() * subpibs[idx, range(lng), 5+a_obs[:lng] // 5].cpu().numpy())
        ir[idx, lng:] = 1  # Mask out the padded timesteps

    weights = np.clip(ir.cumprod(axis=1), 0, 1e3)
    if weighted:
        weights_norm = weights.sum(axis=0)
    else:
        weights_norm = weights.shape[0]
    weights /= weights_norm

    ess = (weights[:,-1].sum()) ** 2 / (((weights[:,-1]) ** 2).sum())
    estm = (weights[:,-1] * discounted_rewards.sum(axis=-1)).sum()

    return estm, ess
