import unittest

import numpy as np
import torch

from pets.pets_models import GraphTransitionModel, GraphTransitionPets
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.models.util as model_util


class MyTestCase(unittest.TestCase):
    def test_something(self):
        env = cartpole_env.CartPoleEnv(render_mode="rgb_array")
        obs, _ = env.reset(0)
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        act = env.action_space.sample()

        obs = model_util.to_tensor(obs).to(device)
        act = model_util.to_tensor(act).to(device)
        model_in = self.compile_obs_action(act, obs)

        pets = GraphTransitionPets(in_size=obs_shape[0]+act_shape[0], out_size=obs_shape[0], device=device, ensemble_size=5)

        x = model_in.unsqueeze(0)
        self.forward_twice(act, pets, x)

        x = x.tile(64, 1)
        self.forward_twice(act, pets, x)

    def forward_twice(self, act, pets, x):
        x2 = pets.forward(x)
        model_in = self.compile_obs_action(act, x2[0])
        pets.forward(model_in)

    def compile_obs_action(self, act, obs):
        shp = np.array(list(obs.shape))
        shp[-len(act.shape):] = 1
        a = act.tile(shp.tolist())
        return torch.cat([obs, a], dim=obs.ndim - 1).float()


if __name__ == '__main__':
    unittest.main()
