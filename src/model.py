import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
from typing import Optional
import matplotlib.pyplot as plt


class UNet(nnx.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: int,
        time_emb_dim: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.features = features
        # accept plain dicts for rngs (e.g. {"params": jax.random.PRNGKey(0)})
        self.rngs = rngs

    def _create_residual_block(
        self, in_channels: int, out_channels: int, rngs: nnx.Rngs
    ):

        conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )

        norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        norm2 = nnx.LayerNorm(out_channels, rngs=rngs)

        shortcut = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs,
        )

        def forward(x: jax.Array) -> jax.Array:
            indentity = shortcut(x)
            x = conv1(x)
            x = norm1(x)
            x = nnx.gelu(x)
            x = conv2(x)
            x = norm2(x)
            x = nnx.gelu(x)
            return x + indentity

    def _pos_encoding(self, t: jax.Array, dim: int) -> jax.Array:
        """Generate sinuisoidal positional encodings."""

        half_dim = dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if dim % 2 == 1:  # zero pad
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb


if __name__ == "__main__":

    unet = UNet(in_channels=1, out_channels=1, features=64, rngs=nnx.Rngs(0))

    # times from 1 to 10
    t_vals = jnp.arange(1, 11)
    # compute positional embeddings (use a moderate dim so formula is stable)
    pos = unet._pos_encoding(t_vals, dim=16)

    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, pos[:, 0], label="component 0")
    plt.plot(t_vals, pos[:, 1], label="component 1")
    plt.plot(t_vals, pos[:, 2], label="component 2")
    plt.plot(t_vals, pos[:, 3], label="component 3")
    plt.plot(t_vals, pos[:, 4], label="component 4")

    plt.xlabel("t")
    plt.ylabel("positional embedding value")
    plt.title("Positional embeddings for t = 1..10 (first five components)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
