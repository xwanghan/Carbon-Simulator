import numpy as np
from gymnasium.spaces import Box, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork, add_time_dimension
from ray.rllib.utils import try_import_torch

# Disable TF INFO, WARNING, and ERROR messages
torch, nn = try_import_torch()

_WORLD_MAP_NAME = "world-map"
_WORLD_IDX_MAP_NAME = "world-idx_map"
_MASK_NAME = "action_mask"


def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = torch.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


class ConvRnn(RecurrentNetwork, nn.Module):
    custom_name = "Conv_Rnn"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        input_emb_vocab = self.model_config["custom_model_config"]["input_emb_vocab"]
        emb_dim = self.model_config["custom_model_config"]["idx_emb_dim"]
        num_conv = self.model_config["custom_model_config"]["num_conv"]
        num_fc = self.model_config["custom_model_config"]["num_fc"]
        self.cell_size = self.model_config["custom_model_config"]["cell_size"]
        generic_name = self.model_config["custom_model_config"].get("generic_name", None)

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        if not isinstance(obs_space, Dict):
            if isinstance(obs_space, Box):
                raise TypeError(
                    "({}) Observation space should be a gymnasium Dict."
                    " Is a Box of shape {}".format(name, obs_space.shape)
                )
            raise TypeError(
                "({}) Observation space should be a gymnasium Dict."
                " Is {} instead.".format(name, type(obs_space))
            )

        # Define input layers
        self.conv_input_keys = []
        self.non_conv_input_keys = []
        self.mask = []
        input_dict = {}
        conv_shape_r = None
        conv_shape_c = None
        conv_map_channels = None
        conv_idx_channels = None
        self.found_world_map = False
        self.found_world_idx = False
        for k, v in obs_space.spaces.items():
            assert v.shape, obs_space.spaces
            shape = v.shape
            input_dict[k] = shape
            if k == _MASK_NAME:
                self.mask.append(k)
            elif k == _WORLD_MAP_NAME:
                conv_shape_r, conv_shape_c, conv_map_channels = (
                    v.shape[0],
                    v.shape[1],
                    v.shape[2],
                )
                self.found_world_map = True
                self.conv_input_keys.append(k)
            elif k == _WORLD_IDX_MAP_NAME:
                conv_idx_channels = v.shape[2] * emb_dim
                self.found_world_idx = True
                self.conv_input_keys.append(k)
            else:
                self.non_conv_input_keys.append(k)

        # Determine which of the inputs are treated as non-conv inputs
        if generic_name is None:
            pass
        elif isinstance(generic_name, (tuple, list)):
            self.non_conv_input_keys = generic_name
        elif isinstance(generic_name, str):
            self.non_conv_input_keys = [generic_name]
        else:
            raise TypeError

        if self.found_world_map:
            assert self.found_world_idx
            self.use_conv = True
            self.conv_shape = (
                conv_shape_r,
                conv_shape_c,
                conv_map_channels,
                conv_idx_channels,
            )

        else:
            assert not self.found_world_idx
            self.use_conv = False
            self.conv_shape = None

        assert self.use_conv

        self.word_idx_embedding = nn.Embedding(input_emb_vocab, emb_dim)

        conv_layers = [
            nn.Conv2d(self.conv_shape[2] + self.conv_shape[3], 16, (3, 3), 2),
            nn.ReLU(),
        ]
        for i in range(num_conv - 1):
            conv_layers += [
                nn.Conv2d(16, 16, (3, 3), 2),
                nn.ReLU()
            ]
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)

        dummy_in = torch.ones((1, self.conv_shape[2] + self.conv_shape[3], self.conv_shape[0], self.conv_shape[1]))
        dummy_out = self.conv(dummy_in)
        self.conv_out_size = dummy_out.shape
        self.fc_in_size = self.conv_out_size[1]
        for k in self.non_conv_input_keys:
            self.fc_in_size += input_dict[k][0]

        fc_layers = [
            nn.Linear(self.fc_in_size, self.cell_size),
            nn.ReLU(),
        ]
        for i in range(num_fc - 1):
            fc_layers += [
                nn.Linear(self.cell_size, self.cell_size),
                nn.ReLU(),
            ]
        fc_layers.append(nn.LayerNorm([self.cell_size]))
        self.fc = nn.Sequential(*fc_layers)

        self.lstm = nn.LSTM(self.cell_size, self.cell_size, 1)

        self.num_outputs = num_outputs

        self._logits_branch = nn.Linear(self.cell_size, self.num_outputs)

        self._value_branch = nn.Linear(self.cell_size, 1)

        self._value_out = None

    def _extract_input_list(self, dictionary):
        return [dictionary[k] for k in self.non_conv_input_keys]

    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""

        non_conv_input = torch.cat(self._extract_input_list(input_dict["obs"]), dim=-1)

        value1 = input_dict["obs"][_WORLD_MAP_NAME]

        value2 = input_dict["obs"][_WORLD_IDX_MAP_NAME]

        value2 = self.word_idx_embedding(value2.long())
        value2 = value2.reshape((-1, self.conv_shape[0], self.conv_shape[1], self.conv_shape[3]))

        conv_in = torch.cat((value1, value2), dim=-1).permute(0, 3, 1, 2)
        conv_out = self.conv(conv_in)

        # assert conv_out.shape[-1] == self.conv_out_size[1], (conv_out.shape, self.conv_out_size, self.conv_shape)
        fc_in = torch.cat((non_conv_input, conv_out), dim=-1)

        fc_out = self.fc(fc_in)

        rnn_in = fc_out.reshape(value1.shape[0] // seq_lens.shape[0], seq_lens.shape[0], self.cell_size)

        rnn_out, [h, c] = self.lstm(rnn_in, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])

        logit_in = rnn_out.reshape(-1, self.cell_size)

        self._value_out = self._value_branch(logit_in)

        model_out = self._logits_branch(logit_in)

        model_out = apply_logit_mask(model_out, input_dict["obs"][_MASK_NAME])

        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def get_initial_state(self):
        return [torch.zeros(self.cell_size).float(),
                torch.zeros(self.cell_size).float(),
                ]

    def value_function(self):
        return torch.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model(ConvRnn.custom_name, ConvRnn)
