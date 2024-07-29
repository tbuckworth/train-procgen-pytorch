import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import einops

import mbrl.util.math

from mbrl.models import Ensemble, truncated_normal_init

class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True, first: bool = False
    ):
        super().__init__()
        self.first = first
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            w = self.weight[self.elite_models, ...]
            b = self.bias[self.elite_models, ...]
        else:
            w = self.weight
            b = self.bias
        # First ensemble linear layer expands dimensions, rest leave dims alone
        try:
            xw = einops.einsum(x, w, "e ... d, e d h -> e ... h")
        except RuntimeError as e:
            xw = einops.einsum(x, w, "... d, e d h -> e ... h")
        if self.use_bias:
            shp = np.array(list(xw.shape))
            shp[1:-1] = 1
            try:
                return xw + b.reshape(shp.tolist())
            except RuntimeError as e:
                raise(e)
        return xw

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite



class MLPModel(nn.Module):
    def __init__(self, linear_cons, in_channels, depth, mid_weight, latent_size, is_first=False):
        super(MLPModel, self).__init__()
        self.input_size = in_channels
        self.depth = depth
        self.mid_weight = mid_weight
        self.output_dim = latent_size
        mid_layers = []
        for _ in range(depth - 2):
            mid_layers.append(linear_cons(self.mid_weight, self.mid_weight))
            mid_layers.append(nn.ReLU())

        self.model = nn.Sequential(
            linear_cons(self.input_size, self.mid_weight, first=is_first),
            nn.ReLU(),
            nn.Sequential(*mid_layers),
            nn.LayerNorm(self.mid_weight),
            linear_cons(self.mid_weight, self.output_dim),
        )

    def forward(self, x):
        return self.model(x)

class GraphTransitionModel(nn.Module):
    def __init__(self, linear_cons, in_channels, depth, mid_weight, latent_size, device, deterministic=False):
        super(GraphTransitionModel, self).__init__()
        self.input_size = in_channels
        self.depth = depth
        self.mid_weight = mid_weight
        self.output_dim = latent_size
        self.device = device

        self.messenger = MLPModel(linear_cons, 5, depth, mid_weight, latent_size, is_first=True)
        if not deterministic:
            latent_size *= 2
        self.updater = MLPModel(linear_cons, 3, depth, mid_weight, latent_size)

    def concater(self, x, y, axis):
        return torch.concat([x.unsqueeze(axis), y.unsqueeze(axis)], axis=axis)

    def update(self, h, y):
        tile_shape = [1 for _ in y.shape] + [1]
        if h.shape[0] != y.shape[0]:
            tile_shape[0] = y.shape[0]
        x = h.tile(tile_shape)
        x2 = torch.concat([x, y.unsqueeze(-1)], -1)
        return self.updater(x2)

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            print("huh?")
        obs = x[..., :-1]
        action = x[..., -1]
        # Normalize actions?
        n, h = self.prep_input(obs)
        msg = self.sum_all_messages(n, h, action)
        return self.update(h, msg).squeeze(dim=-1)

    def prep_input(self, obs):
        x = self.append_index(obs.squeeze(dim=-1))
        n = x.shape[-2]
        return n, x

    def sum_all_messages(self, n, x, action):
        msg_in = self.vectorize_for_message_pass(action, n, x)
        messages = self.messenger(msg_in)
        return torch.sum(messages, dim=-2).squeeze(dim=-1)

    def vectorize_for_message_pass(self, action, n, x):
        xi = x.unsqueeze(-2).tile([n, 1])
        xj = x.unsqueeze(-3).tile([n, 1, 1])
        a = action.unsqueeze(-1).unsqueeze(-1).tile([n, n]).unsqueeze(-1)
        try:
            msg_in = torch.concat([xi, xj, a], dim=-1)
        except RuntimeError as e:
            raise(e)
            print(e)
        return msg_in

    def vec_for_update(self, messages, x):
        msg = torch.sum(messages, dim=-2).squeeze()
        h = torch.concat([x, msg.unsqueeze(-1)], -1)
        u = self.updater(h)
        return h, u

    def append_index(self, x):
        n = x.shape[-1]
        coor = torch.FloatTensor([i / n for i in range(n)])
        shp = [i for i in x.shape[:-1]] + [1]
        all_coor = torch.tile(coor, shp).to(device=self.device)
        return self.concater(x, all_coor, -1)


class GraphTransitionPets(Ensemble):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            device: Union[str, torch.device],
            num_layers: int = 4,
            ensemble_size: int = 1,
            hid_size: int = 200,
            deterministic: bool = False,
            propagation_method: Optional[str] = None,
            learn_logvar_bounds: bool = False,
            activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        self.in_size = in_size
        self.out_size = out_size

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out, first=False):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out, first=first)

        # hidden_layers = [
        #     nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        # ]
        # for i in range(num_layers - 1):
        #     hidden_layers.append(
        #         nn.Sequential(
        #             create_linear_layer(hid_size, hid_size),
        #             create_activation(),
        #         )
        #     )
        # self.hidden_layers = nn.Sequential(*hidden_layers)

        self.hidden_layers = GraphTransitionModel(create_linear_layer, self.in_size, num_layers, hid_size, 1, self.device, deterministic)

        if not deterministic:
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )

        self.apply(truncated_normal_init)
        self.to(self.device)

        self.elite_models: List[int] = None

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            self.hidden_layers.messenger.set_elite(self.elite_models)
            self.hidden_layers.messenger.toggle_use_only_elite()
            self.hidden_layers.updater.set_elite(self.elite_models)
            self.hidden_layers.updater.toggle_use_only_elite()

    def _default_forward(
            self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        mean_and_logvar = self.hidden_layers(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            return mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., 0]
            logvar = mean_and_logvar[..., 1]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            return mean, logvar

    def _forward_from_indices(
            self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
            self,
            x: torch.Tensor,
            rng: Optional[torch.Generator] = None,
            propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
            self,
            x: torch.Tensor,
            rng: Optional[torch.Generator] = None,
            propagation_indices: Optional[torch.Tensor] = None,
            use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        # if target.shape[0] != self.num_members:
        n_dims = len(pred_mean.shape) - len(target.shape)
        if n_dims > 0:
            shp = np.array(pred_mean.shape)
            shp[n_dims:] = 1
            target = target.repeat(*shp)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
            self,
            model_in: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        return self._nll_loss(model_in, target), {}

    def eval_score(  # type: ignore
            self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    def sample_propagation_indices(
            self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]
