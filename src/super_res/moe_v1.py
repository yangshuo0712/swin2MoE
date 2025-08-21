import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from copy import deepcopy
import numpy as np

from .utils import Mlp as MLP


class LeFFExpert(nn.Module):
    """
    Locally Enhanced Feed-Forward expert.

    This expert expands the token features into a higher dimensional space,
    applies a depthwise convolution along the sequence dimension to capture
    local context, then projects back to the original embedding dimension.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # use a depthwise 1‑D convolution over the token dimension.  We use
        # groups=hidden_dim to perform per‑channel convolutions.
        self.dwconv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (N, embed_dim) – sequence of tokens.  The input is assumed
               to be a flattened sequence; no spatial shape is required.

        Returns:
            Tensor of shape (N, embed_dim) after locally enhanced feed‑forward.
        """
        # project up
        x = self.fc1(x)
        # reshape to (batch=1, channels=hidden_dim, seq_len=N) for depthwise conv
        x1 = x.transpose(0, 1).unsqueeze(0)  # (1, hidden_dim, N)
        x1 = self.dwconv(x1)
        x = x1.squeeze(0).transpose(0, 1)    # (N, hidden_dim)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.

    The purpose of this class is to create input mini‑batches for the experts
    and to combine the results of the experts to form a unified output tensor.

    There are two functions: dispatch – take an input Tensor and create input
    Tensors for each expert; combine – take output Tensors from each expert and
    form a combined output Tensor.  Outputs from different experts for the same
    batch element are summed together, weighted by the provided gates.
    """

    def __init__(self, num_experts: int, gates: torch.Tensor):
        """Create a SparseDispatcher.

        Args:
            num_experts: Number of experts.
            gates: A tensor of shape [batch_size, num_experts] containing the
                gate weights.  Batch element b is sent to expert e iff
                gates[b, e] != 0.
        """
        self._gates = gates
        self._num_experts = num_experts
        # determine the batch index for each expert
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate how many samples each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match the flattened expert inputs
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp: torch.Tensor):
        """Create one input Tensor for each expert.

        Args:
            inp: a tensor of shape [batch_size, input_size]

        Returns:
            a list of `num_experts` tensors with shapes [expert_batch_size_i, input_size]
        """
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates: bool = True, cnn_combine=None):
        """Sum together the expert output, weighted by the gates.

        The slice corresponding to a particular batch element b is computed as
        the sum over all experts i of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False,
        the gate values are ignored.

        Args:
            expert_out: a list of `num_experts` tensors, each with shape
                [expert_batch_size_i, output_size].
            multiply_by_gates: whether to weight the outputs by the gate values.
            cnn_combine: if provided, apply this 2‑D convolution to combine
                outputs from multiple experts (Smart Merger).

        Returns:
            a tensor with shape [batch_size, output_size].
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
        zeros = torch.zeros(
            (self._gates.size(0),) + expert_out[-1].shape[1:],
            device=stitched.device,
            dtype=stitched.dtype,
        )
        if cnn_combine is not None:
            return self.smartly_combine(stitched, cnn_combine)
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def smartly_combine(self, stitched, cnn_combine):
        """Apply a convolutional combine (Smart Merger) across active experts.

        This method groups the outputs belonging to the same sample and stacks
        them along a new dimension to form a tensor of shape
        (batch_size, num_active_experts, output_size).  The convolution is
        applied across the num_active_experts dimension and outputs are merged.
        """
        idxes = []
        for i in self._batch_index.unique():
            idx = (self._batch_index == i).nonzero().squeeze(1)
            idxes.append(idx)
        idxes = torch.stack(idxes)
        # apply the convolution; input shape becomes (batch_size, num_active, output_size)
        return cnn_combine(stitched[idxes]).squeeze(1)

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per‑expert tensors.

        Returns:
            a list of `num_experts` one‑dimensional tensors with shapes
            [expert_batch_size_i]
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


def build_experts(experts_cfg, default_cfg, num_experts):
    """Build a list of experts according to the configuration.

    Args:
        experts_cfg: list of dictionaries specifying each expert.  Each entry
            may include the keys 'type' and 'hid_ratio'.  Supported types are
            'mlp' and 'leff'.  If None, a homogeneous list of MLP experts is
            created.
        default_cfg: tuple (input_dim, hidden_dim, output_dim) specifying the
            base dimensions for experts.
        num_experts: total number of experts to build when experts_cfg is None.

    Returns:
        a nn.ModuleList of experts.
    """
    experts_cfg = deepcopy(experts_cfg)
    if experts_cfg is None:
        return nn.ModuleList([MLP(*default_cfg) for _ in range(num_experts)])

    experts = []
    for e_cfg in experts_cfg:
        type_ = e_cfg.get("type", "mlp")
        hid_ratio = e_cfg.get("hid_ratio", 1.0)
        in_dim, hid_dim, out_dim = default_cfg
        if type_ == "mlp":
            # scale hidden size if hid_ratio provided
            hdim = int(hid_dim * hid_ratio)
            experts.append(MLP(in_dim, hdim, out_dim))
        elif type_ == "leff":
            hdim = int(hid_dim * hid_ratio)
            experts.append(LeFFExpert(in_dim, hdim))
        else:
            raise ValueError(f"Unknown expert type: {type_}")
    return nn.ModuleList(experts)


class MoE(nn.Module):
    """Sparsely gated Mixture of Experts layer.

    This implementation supports heterogeneous experts (MLP or LeFF) and
    configurable gating strategies.  It also implements an optional dynamic
    routing mechanism to decide how many experts to activate per sample.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_experts: int,
        experts=None,
        noisy_gating: bool = True,
        k: int = 4,
        x_gating = None,
        with_noise: bool = True,
        with_smart_merger = None,
        dynamic_route: bool = False,
        conf_threshold: float = 0.0,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        assert self.k <= self.num_experts
        # instantiate experts
        self.experts = build_experts(
            experts, (self.input_size, self.hidden_size, self.output_size), num_experts
        )
        # gating parameters
        self.w_gate = nn.Parameter(
            torch.zeros(input_size, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_size, num_experts), requires_grad=True
        )
        self.x_gating = x_gating  # 'global', 'spatial', 'conv'
        if self.x_gating in ("conv", "conv1d"):
            # a simple 1‑D conv over the token dimension to extract gating features
            # input channels = 1 (token features are averaged), output channels = 1
            self.x_gate = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        else:
            self.x_gate = None
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        # smart merger support
        self.cnn_combine = None
        if with_smart_merger == "v1":
            # convolution merges K expert outputs into one; K = self.k by default
            self.cnn_combine = nn.Conv2d(self.k, 1, kernel_size=3, padding=1)
        # dynamic routing parameters
        self.dynamic_route = dynamic_route
        self.conf_threshold = conf_threshold

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the squared coefficient of variation of a sample."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute the true load per expert, given the gates."""
        return (gates > 0).sum(0)

    def noisy_top_k_gating(self, xg: torch.Tensor, train: bool, noise_epsilon: float = 1e-2):
        """Noisy top‑k gating with optional dynamic routing.

        Args:
            xg: aggregated features used for gating, shape [batch_size, input_size].
            train: whether the model is in training mode.
            noise_epsilon: small constant added to noise.

        Returns:
            gates: tensor of shape [batch_size, num_experts] with gate probabilities.
            load: tensor of shape [num_experts] counting usage.
        """
        use_amp = torch.is_autocast_enabled()
        # compute logits
        if use_amp:
            with torch.autocast(device_type="cuda", enabled=False):
                x_f32 = xg.float()
                clean_logits = x_f32 @ self.w_gate.float()
                if self.noisy_gating and train:
                    raw_noise_stddev = x_f32 @ self.w_noise.float()
                    noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
                    noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
                    logits = noisy_logits
                else:
                    logits = clean_logits
        else:
            x_f32 = xg.float()
            clean_logits = x_f32 @ self.w_gate.float()
            if self.noisy_gating and train:
                raw_noise_stddev = x_f32 @ self.w_noise.float()
                noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
                noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
                logits = noisy_logits
            else:
                logits = clean_logits

        # select top‑k+1 for potential use
        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts), dim=1
        )
        top_k_logits = top_logits[:, : self.k].clone()
        top_k_indices = top_indices[:, : self.k].clone()

        if self.dynamic_route and self.k >= 2:
            # compute confidence score as difference between best and second best logits
            conf_score = top_k_logits[:, 0] - top_k_logits[:, 1]
            # mask where confidence is above threshold
            use_one = conf_score > self.conf_threshold
            if use_one.any():
                # set the second logit very negative for these samples so its softmax ~0
                top_k_logits[use_one, 1] = -1e9
                # also copy the first index to the second position so dispatcher logic sees same expert
                top_k_indices[use_one, 1] = top_k_indices[use_one, 0]

        # softmax over the selected logits
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # compute load
        if self.noisy_gating and self.k < self.num_experts and train:
            # fall back to deterministic load for simplicity
            load = self._gates_to_load(gates)
        else:
            load = self._gates_to_load(gates)

        return gates.to(xg.dtype), load.to(xg.dtype)

    def forward(self, x: torch.Tensor, loss_coef: float = 1e-2):
        """
        Args:
            x: tensor shape [batch_size, seq_len, input_size]
            loss_coef: multiplier for the load‑balancing loss.

        Returns:
            y: tensor of shape [batch_size, seq_len, output_size]
            loss: scalar load‑balancing loss.
        """
        # compute gating input
        # x arrives as [batch, seq_len, embed_dim]
        if self.x_gating is None or self.x_gating == "global":
            # global average over tokens
            xg = x.mean(dim=1)
        elif self.x_gating == "spatial":
            # weighted average based on token variance
            texture = x.var(dim=-1)
            weights = texture / (texture.sum(dim=1, keepdim=True) + 1e-6)
            xg = (x * weights.unsqueeze(-1)).sum(dim=1)
        elif self.x_gating in ("conv", "conv1d"):
            # simple 1‑D conv along sequence dimension
            x_mean = x.mean(dim=2, keepdim=True)  # (batch, seq_len, 1)
            x_for_conv = x_mean.permute(0, 2, 1)
            conv_out = self.x_gate(x_for_conv)
            xg = conv_out.squeeze(1).mean(dim=1, keepdim=False)
        else:
            # fallback to global average
            xg = x.mean(dim=1)

        # obtain gates and load
        gates, load = self.noisy_top_k_gating(
            xg, self.training and self.noisy_gating
        )

        # calculate importance and load balancing loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = loss * loss_coef

        # prepare dispatcher
        # flatten x for expert consumption: [batch*seq_len, input_size]
        B, S, C = x.shape
        x_flat = x.reshape(B * S, C)
        # repeat gates for each token
        repeat_gates = gates.repeat_interleave(S, dim=0)
        dispatcher = SparseDispatcher(self.num_experts, repeat_gates)

        expert_inputs = dispatcher.dispatch(x_flat)
        gate_list = dispatcher.expert_to_gates()
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            if expert_inputs[i].shape[0] == 0:
                expert_outputs.append(
                    torch.zeros(
                        (0, self.output_size),
                        device=x.device,
                        dtype=x.dtype,
                    )
                )
                continue
            expert_outputs.append(expert(expert_inputs[i]))
        # combine outputs
        y_flat = dispatcher.combine(
            expert_outputs, cnn_combine=self.cnn_combine
        )
        # reshape back to [batch, seq_len, output_size]
        y = y_flat.reshape(B, S, self.output_size)
        return y, loss
