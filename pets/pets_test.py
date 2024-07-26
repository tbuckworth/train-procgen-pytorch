import unittest

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
        model_in = torch.cat([obs, act], dim=obs.ndim - 1).float()

        pets = GraphTransitionPets(in_size=obs_shape[0]+act_shape[0], out_size=obs_shape[0], device=device)

        x = model_in.unsqueeze(0)
        pets.forward(x)

        x = x.tile(64, 1)
        pets.forward(x)


if __name__ == '__main__':
    unittest.main()
