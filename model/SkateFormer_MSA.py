import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


# -------------------------------
# Helper Functions and Modules
# -------------------------------

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Truncated normal initialization.
    """

    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        raise ValueError("mean is too far from [a, b] for truncation to be effective.")

    with torch.no_grad():
        # Get upper and lower cdf
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u]
        tensor.uniform_(l, u)

        # Convert to standard normal
        tensor.sub_(l).div_(u - l)
        # Approximate inverse CDF using erfinv
        tensor = tensor.erfinv() * math.sqrt(2.)

        # Clamp to ensure it's within [a, b]
        tensor.clamp_(min=(a - mean) / std, max=(b - mean) / std)

        # Scale and shift
        tensor.mul_(std).add_(mean)
    return tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Work with tensor of shape [batch, ...]
    shape = (x.shape[0],) + (1,) * (x.ndimension() - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Mlp(nn.Module):
    """
    Multi-layer perceptron with one hidden layer.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------------
# Dynamic Partition Module
# -------------------------------

class DynamicPartitionMLP(nn.Module):
    """
    A small MLP to dynamically generate partition sizes based on input dimensions.
    """

    def __init__(self, hidden_dim=64, max_size_t=8, max_size_v=8):
        super(DynamicPartitionMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: [partition_size_t, partition_size_v]
        )
        self.max_size_t = max_size_t
        self.max_size_v = max_size_v

    def forward(self, T, V):
        """
        Input:
            T: int, time dimension
            V: int, space dimension
        Output:
            partition_size: tuple of two ints
        """
        device = next(self.parameters()).device
        in_feat = torch.tensor([float(T), float(V)], device=device).unsqueeze(0)  # Shape: [1, 2]
        out_feat = self.encoder(in_feat)  # Shape: [1, 2]
        partition_size = F.relu(out_feat)  # Ensure positive
        partition_size = torch.clamp(partition_size, min=1,
                                     max=torch.tensor([self.max_size_t, self.max_size_v], device=device,
                                                      dtype=torch.float32))
        partition_size = partition_size.round().long().squeeze(0)  # Shape: [2]
        return partition_size  # [partition_size_t, partition_size_v]


# -------------------------------
# Partition and Reverse Functions
# -------------------------------

def type_1_partition(input, partition_size):  # partition_size = [N, L]
    B, C, T, V = input.shape
    N, L = partition_size
    assert T % N == 0 and V % L == 0, "T and V must be divisible by partition sizes."
    partitions = input.view(B, C, T // N, N, V // L, L)
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, N, L, C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    N, L = partition_size
    B = int(partitions.shape[0] / (T * V / N / L))
    output = partitions.view(B, T // N, V // L, N, L, -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    N, K = partition_size
    assert T % N == 0 and V % K == 0, "T and V must be divisible by partition sizes."
    partitions = input.view(B, C, T // N, N, K, V // K)
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, N, K, C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    N, K = partition_size
    B = int(partitions.shape[0] / (T * V / N / K))
    output = partitions.view(B, T // N, V // K, N, K, -1)
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    M, L = partition_size
    assert T % M == 0 and V % L == 0, "T and V must be divisible by partition sizes."
    partitions = input.view(B, C, M, T // M, V // L, L)
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, M, L, C)
    return partitions


def type_3_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    M, L = partition_size
    B = int(partitions.shape[0] / (T * V / M / L))
    output = partitions.view(B, T // M, V // L, M, L, -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    M, K = partition_size
    assert T % M == 0 and V % K == 0, "T and V must be divisible by partition sizes."
    partitions = input.view(B, C, M, T // M, K, V // K)
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, M, K, C)
    return partitions


def type_4_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    M, K = partition_size
    B = int(partitions.shape[0] / (T * V / M / K))
    output = partitions.view(B, T // M, V // K, M, K, -1)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output


# -------------------------------
# Relative Position Index
# -------------------------------

def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T, device='cpu')], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)  # Shape: [T, T]


# -------------------------------
# Multi-Head Self Attention with Relative Positional Bias
# -------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size  # Tuple: (partition_size_t, partition_size_v)
        self.scale = (num_heads) ** -0.5
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel

        if self.rel:
            if self.rel_type in ['type_1', 'type_3']:
                # Relative position bias for type_1 and type_3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), num_heads))  # [2N-1, H]
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)
            elif self.rel_type in ['type_2', 'type_4']:
                # Relative position bias for type_2 and type_4
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1],
                                num_heads))  # [2N-1, K, K, H]
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=True)
        self.attn_proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=attn_drop)

    def _get_relative_positional_bias(self, device):
        if self.rel_type in ['type_1', 'type_3']:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            relative_position_bias = relative_position_bias.view(
                self.partition_size[0], self.partition_size[0], -1
            )  # [N, N, H]
            # Expand to [N*L, N*L, H] by repeating for spatial partitions
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3)  # [N,1,N,1,H]
            relative_position_bias = relative_position_bias.repeat(1, 1, 1, 1, 1)  # Modify as needed for multi-scale
            relative_position_bias = relative_position_bias.view(-1, -1, self.num_heads)  # [N*1, N*1, H]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [H, N, N]
            return relative_position_bias.unsqueeze(0).to(device)  # [1, H, N, N]
        elif self.rel_type in ['type_2', 'type_4']:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            relative_position_bias = relative_position_bias.view(
                self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1
            )  # [N, N, K, K, H]
            # Permute and reshape to [N*K, N*K, H]
            relative_position_bias = relative_position_bias.permute(0, 2, 1, 3, 4).contiguous()
            relative_position_bias = relative_position_bias.view(
                self.partition_size[0] * self.partition_size[1],
                self.partition_size[0] * self.partition_size[1],
                -1
            )  # [N*K, N*K, H]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [H, N*K, N*K]
            return relative_position_bias.unsqueeze(0).to(device)  # [1, H, N*K, N*K]

    def forward(self, input):
        """
        Input:
            input: [B*N*L, S, C] where S = N*L
        Output:
            output: [B*N*L, S, C]
        """
        B_, S, C = input.shape
        device = input.device
        qkv = self.qkv(input).reshape(B_, S, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [3, B_, H, S, C//H]
        q, k, v = qkv.unbind(0)  # Each has shape [B_, H, S, C//H]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, H, S, S]

        if self.rel:
            relative_position_bias = self._get_relative_positional_bias(device)  # [1, H, S, S]
            attn = attn + relative_position_bias

        attn = self.softmax(attn)  # [B_, H, S, S]
        attn = self.attn_drop(attn)
        output = (attn @ v)  # [B_, H, S, C//H]
        output = output.transpose(1, 2).reshape(B_, S, C)
        output = self.attn_proj(output)
        output = self.proj_drop(output)
        return output


# -------------------------------
# SkateFormerBlock with Dynamic Partition
# -------------------------------

class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, num_points=16, kernel_size=7, num_heads=8,
                 type_1_max_size=(4, 4), type_2_max_size=(4, 4),
                 type_3_max_size=(4, 4), type_4_max_size=(4, 4),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SkateFormerBlock, self).__init__()
        self.type_1_partition = DynamicPartitionMLP(max_size_t=type_1_max_size[0], max_size_v=type_1_max_size[1])
        self.type_2_partition = DynamicPartitionMLP(max_size_t=type_2_max_size[0], max_size_v=type_2_max_size[1])
        self.type_3_partition = DynamicPartitionMLP(max_size_t=type_3_max_size[0], max_size_v=type_3_max_size[1])
        self.type_4_partition = DynamicPartitionMLP(max_size_t=type_4_max_size[0], max_size_v=type_4_max_size[1])

        self.partition_functions = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_functions = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.rel_types = ['type_1', 'type_2', 'type_3', 'type_4']

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)

        # G-Conv parameters
        self.gconv = nn.Parameter(torch.zeros(num_heads // (2 * 2), num_points, num_points))
        trunc_normal_(self.gconv, std=.02)
        self.tconv = nn.Conv2d(in_channels // (2 * 2), in_channels // (2 * 2), kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0), groups=num_heads // (2 * 2))

        # Attention layers
        attention = []
        for i in range(4):
            attention.append(
                MultiHeadSelfAttention(in_channels=in_channels // (4 * 2),
                                       rel_type=self.rel_types[i],
                                       num_heads=num_heads // (4 * 2),
                                       partition_size=(1, 1),  # Placeholder, will be updated dynamically
                                       attn_drop=attn_drop, rel=rel)
            )
        self.attention = nn.ModuleList(attention)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

        self.alpha_attn = nn.Parameter(torch.ones(4))  # 4 个 Attention
        self.alpha_GConv = nn.Parameter(torch.ones(1))
        self.alpha_TConv = nn.Parameter(torch.ones(1))

    def forward(self, input):
        """
        Input:
            input: [B, C, T, V]
        Output:
            output: [B, C, T, V]
        """
        B, C, T, V = input.shape

        # Partition
        input = input.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()  # [B, 2C, T, V]

        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2],
                                     dim=1)  # f_conv: [B, C//2, T, V]; f_attn: [B, 3C//2, T, V]
        y = []

        # G-Conv
        split_f_conv = torch.chunk(f_conv, 2, dim=1)  # Two parts for G-Conv and T-Conv
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)  # Split for each G-Conv kernel
        for i in range(self.gconv.shape[0]):
            if self.gconv.shape[-1] != V:
                raise ValueError(f"gconv dimension {self.gconv.shape[-1]} does not match V={V}")
            z = torch.einsum('nctu,vu->nctv', split_f_gconv[i], self.gconv[i])  # [B, C', T, V]
            y_gconv.append(z)
        y.append(torch.cat(y_gconv, dim=1) * self.alpha_GConv)  # [B, C//2, T, V]

        # T-Conv
        y.append(self.tconv(split_f_conv[1]) * self.alpha_TConv)  # [B, C//2, T, V]

        # Skate-MSA
        split_f_attn = torch.chunk(f_attn, 4, dim=1)  # Assuming 4 partition types

        for i in range(4):
            # Dynamically get partition size
            dynamic_partition = self.partition_functions[i]
            partition_size = dynamic_partition(T, V)  # [partition_size_t, partition_size_v]

            # Ensure partition sizes divide T and V
            partition_size_t, partition_size_v = partition_size
            if T % partition_size_t != 0 or V % partition_size_v != 0:
                raise ValueError(f"Partition sizes {partition_size} do not divide T={T} or V={V}")

            # Update the partition_size in the attention module
            self.attention[i].partition_size = (partition_size_t.item(), partition_size_v.item())

            # Update relative_position_index buffer
            if self.attention[i].rel:
                new_rel_index = get_relative_position_index_1d(partition_size_t.item()).to(input.device)
                self.attention[i].relative_position_index = new_rel_index

                # Re-initialize the relative_position_bias_table if partition_size changes
                if self.attention[i].rel_type in ['type_1', 'type_3']:
                    # Adjust relative_position_bias_table size
                    self.attention[i].relative_position_bias_table.data = nn.init.trunc_normal_(
                        self.attention[i].relative_position_bias_table, std=.02
                    )
                elif self.attention[i].rel_type in ['type_2', 'type_4']:
                    self.attention[i].relative_position_bias_table.data = nn.init.trunc_normal_(
                        self.attention[i].relative_position_bias_table, std=.02
                    )

            # Partition
            partition_func = [type_1_partition, type_2_partition, type_3_partition, type_4_partition][i]
            reverse_func = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse][i]
            input_partitioned = partition_func(split_f_attn[i], partition_size)  # [B*P, N, L, C]
            input_partitioned = input_partitioned.view(-1, partition_size_t * partition_size_v,
                                                       split_f_attn[i].shape[1])  # [B*P, S, C]

            # Apply attention
            att_out = self.attention[i](input_partitioned)  # [B*P, S, C]
            # Reverse partition
            att_out = reverse_func(att_out, (T, V), partition_size)  # [B, C', T, V]
            att_out = att_out.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C']
            y.append(self.alpha_attn[i] * att_out)  # [B, T, V, C']

        # Concatenate all paths
        y = torch.cat(y, dim=-1)  # [B, T, V, C]
        y = y.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
        y = self.proj(y)
        y = self.proj_drop(y)
        output = skip + self.drop_path(y)

        # Feed Forward
        output_ffn = self.mlp(self.norm_2(output.permute(0, 2, 3, 1).contiguous()))  # [B, T, V, C]
        output_ffn = output_ffn.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
        output = output + self.drop_path(output_ffn)

        return output


# -------------------------------
# Example Usage
# -------------------------------

if __name__ == "__main__":
    # Example parameters
    batch_size = 2
    channels = 64
    T = 16  # Time dimension
    V = 16  # Space dimension, must match num_points
    num_points = V  # Ensure num_points matches V
    kernel_size = 3
    num_heads = 8
    attn_drop = 0.1
    drop = 0.1
    drop_path = 0.1
    mlp_ratio = 4.

    # Instantiate SkateFormerBlock
    skate_block = SkateFormerBlock(
        in_channels=channels,
        num_points=num_points,  # Set to match V
        kernel_size=kernel_size,
        num_heads=num_heads,
        type_1_max_size=(4, 4),
        type_2_max_size=(4, 4),
        type_3_max_size=(4, 4),
        type_4_max_size=(4, 4),
        attn_drop=attn_drop,
        drop=drop,
        rel=True,
        drop_path=drop_path,
        mlp_ratio=mlp_ratio,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    )

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skate_block = skate_block.to(device)

    # Create dummy input
    input_tensor = torch.randn(batch_size, channels, T, V).to(device)

    # Forward pass
    try:
        output = skate_block(input_tensor)
        print("Input shape:", input_tensor.shape)
        print("Output shape:", output.shape)
    except Exception as e:
        print("Error during forward pass:", e)
