import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        action = ptu.to_numpy(self.forward(ptu.from_numpy(obs)).sample())

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            policy_ac_dist = distributions.Categorical(logits=self.logits_net(obs))
        else:
            # # TODO: define the forward pass for a policy with a continuous action space.
            means = self.mean_net(obs)
            stds = torch.exp(self.logstd)
            stds = torch.diag(stds)
            policy_ac_dist = distributions.MultivariateNormal(means, scale_tril=stds) #.sample().shape
            # batch_mean = self.mean_net(obs)
            # scale_tril = torch.diag(torch.exp(self.logstd))
            # batch_dim = batch_mean.shape[0]
            # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            # policy_ac_dist = distributions.MultivariateNormal(batch_mean, batch_scale_tril)

            # mean = self.mean_net(obs)
            # std = torch.exp(self.logstd)
            # policy_ac_dist = distributions.Normal(mean, std)
        return policy_ac_dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        policy_ac_dist = self.forward(obs)
        #log_prob = policy_ac_dist.log_prob(actions) 

        # if self.discrete:
        #     loss = (F.cross_entropy(log_prob, actions, reduction='none') * advantages).mean()
        # else:
        #     # squarred error loss
        #     loss = (torch.mean((policy_ac_dist.rsample() - actions) ** 2) * advantages).mean() # TODO: double check this


        log_probs = policy_ac_dist.log_prob(actions)

        loss = -(log_probs * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
