import unittest

import numpy as np
import torch

from pets.pets_models import GraphTransitionModel, GraphTransitionPets
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.models.util as model_util


def compile_obs_action(act, obs):
    shp = np.array(list(obs.shape))
    shp[-len(act.shape):] = 1
    a = act.tile(shp.tolist())
    return torch.cat([obs, a], dim=obs.ndim - 1).float()


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = cartpole_env.CartPoleEnv(render_mode="rgb_array")
        obs, _ = cls.env.reset(0)
        obs_shape = cls.env.observation_space.shape
        act_shape = cls.env.action_space.shape
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.act = cls.env.action_space.sample()

        cls.obs = model_util.to_tensor(obs).to(device)
        cls.act = model_util.to_tensor(cls.act).to(device)
        cls.model_in = compile_obs_action(cls.act, cls.obs)

        cls.pets = GraphTransitionPets(in_size=obs_shape[0] + act_shape[0],
                                       out_size=obs_shape[0], device=device,
                                       ensemble_size=5)

    def test_eval(self):
        x = self.model_in.unsqueeze(0)
        t = self.obs.unsqueeze(0)
        self.pets.eval_score(x, t)

        x = x.tile(64, 1)
        t = t.tile(64, 1)
        self.pets.eval_score(x, t)


    def test_forward(self):
        x = self.model_in.unsqueeze(0)
        self.forward_twice(self.act, self.pets, x)

        x = x.tile(64, 1)
        self.forward_twice(self.act, self.pets, x)

    def forward_twice(self, act, pets, x):
        x2 = pets.forward(x)
        model_in = compile_obs_action(act, x2[0])
        pets.forward(model_in)


if __name__ == '__main__':
    unittest.main()
