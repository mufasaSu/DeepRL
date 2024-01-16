import pickle
import argparse
from cs285.agents.cql_agent import CQLAgent

import os

import numpy as np
import torch
from simulator import Environment, SingleGyreFlowField

from cs285.infrastructure import pytorch_util as ptu
import tqdm

import utils_custom
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

MODEL_FILE_NAME = "100K_100Steps"

OBSERVATION_SHAPE = (2, )

FLOW_FIELD = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)

ACTION_TYPE = "discrete"
NUM_ACTIONS = 8 # Number of discrete actions the agent can take
MAGNITUDE = 1 # Magnitude of Action

MAX_STEPS = 30 # Maximum number of steps per episode
THRESHOLD = 1 # Threshold for the distance to the target


class ReplayBuffer_from_file:
    def __init__(self, data_dict):
        
        self.observations = np.array(data_dict["state"])
        self.actions = np.array(data_dict["action"])
        self.next_observations = np.array(data_dict["next_state"])
        self.rewards = np.array(data_dict["reward"])
        self.dones = np.array(data_dict["done"])
        self.size = len(self.dones)

        assert len(self.observations) == len(self.next_observations)


    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the environment
    start = [4.0, 4.0]
    target = [17.5, 17.5]
    print("Start:", start)
    print("Target:", target)

    env = Environment(FLOW_FIELD, list(start), target, threshold=THRESHOLD,
                      action_type=ACTION_TYPE, num_actions=NUM_ACTIONS, magnitude=MAGNITUDE)
    state = env.reset()

    #assert discrete, "DQN only supports discrete action spaces"

    agent = CQLAgent(OBSERVATION_SHAPE, NUM_ACTIONS, **config["agent_kwargs"])
    ep_len = MAX_STEPS

    
    with open(os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl"), 'rb') as f:
        file = pickle.load(f)

    
    dataset = ReplayBuffer_from_file(file)


    print("Batch size: ", config["batch_size"])
    for step in tqdm.trange(config["training_steps"], dynamic_ncols=True):
        # Train with offline RL
        batch = dataset.sample(config["batch_size"])

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            for k, v in metrics.items():
                logger.log_scalar(v, k, step)
        
        if step % args.eval_interval == 0:
             # Evaluate
            trajectories = utils_custom.sample_n_trajectories(
                 env,
                 agent,
                 args.num_eval_trajectories,
                 ep_len,
             )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str,
                        default="experiments/offline/ocean_simulator_cql.yaml")

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    # -----------------------------------------------------------------------------------
    parser.add_argument("--observation_shape", type=int, nargs="+", default=[2])
    parser.add_argument("--num_actions", type=int, default=NUM_ACTIONS)
    # -----------------------------------------------------------------------------------

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--dataset_dir", type=str, default="datasets/", )

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "final_project_cql_"

    config = make_config(args.config_file)

    logger = make_logger(logdir_prefix, config)

    trained_agent = run_training_loop(config, logger, args)
    torch.save(trained_agent.critic.state_dict(), f'trained_models/{MODEL_FILE_NAME}.pth')
    print(f"Agent saved to trained_models/{MODEL_FILE_NAME}.pth")


if __name__ == "__main__":
    main()

