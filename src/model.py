import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
from typing import Tuple, Callable, List, Optional
import matplotlib.pyplot as plt
from data import load_mnist


class UNet(nnx.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 features: int,
                 time_emb_dim: int = 128,
                 *,
                 rngs: nnx.Rngs):
        """
        Initialize the U-Net architecture with time embedding.
        """
        self.features = features

        # Time embedding layers for diffusion timestep conditioning.
        self.time_mlp_1 = nnx.Linear(in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs)
        self.time_mlp_2 = nnx.Linear(in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs)

        # Time projection layers for different scales.
        self.time_proj1 = nnx.Linear(in_features=time_emb_dim, out_features=features, rngs=rngs)
        self.time_proj2 = nnx.Linear(in_features=time_emb_dim, out_features=features * 2, rngs=rngs)
        self.time_proj3 = nnx.Linear(in_features=time_emb_dim, out_features=features * 4, rngs=rngs)

        # The encoder path.
        self.down_conv1 = self._create_residual_block(in_channels, features, rngs)
        self.down_conv2 = self._create_residual_block(features, features * 2, rngs)
        self.down_conv3 = self._create_residual_block(features * 2, features * 4, rngs)

        # The bridge connecting the encoder and the decoder.
        self.bridge = self._create_residual_block(features * 4, features * 8, rngs)

        # Decoder path with skip connections.
        self.up_conv3 = self._create_residual_block(features * 12, features * 4, rngs)
        self.up_conv2 = self._create_residual_block(features * 6, features * 2, rngs)
        self.up_conv1 = self._create_residual_block(features * 3, features, rngs)

        # Output layers.
        self.final_norm = nnx.LayerNorm(features, rngs=rngs)
        self.final_conv = nnx.Conv(in_features=features,
                                 out_features=out_channels,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding=((1, 1), (1, 1)),
                                 rngs=rngs)

    def _create_residual_block(self,
                              in_channels: int,
                              out_channels: int,
                              rngs: nnx.Rngs) -> Callable:
        """Creates a residual block with two convolutions and normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX PRNG keys.

        Returns:
            Callable: A function that represents the forward pass through the residual block.
        """

        # Convolutional layers with layer normalization.
        conv1 = nnx.Conv(in_features=in_channels,
                        out_features=out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=((1, 1), (1, 1)),
                        rngs=rngs)
        norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        conv2 = nnx.Conv(in_features=out_channels,
                        out_features=out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=((1, 1), (1, 1)),
                        rngs=rngs)
        norm2 = nnx.LayerNorm(out_channels, rngs=rngs)

        # Projection shortcut if dimensions change.
        shortcut = nnx.Conv(in_features=in_channels,
                            out_features=out_channels,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            rngs=rngs)

        # The forward pass through the residual block.
        def forward(x: jax.Array) -> jax.Array:
            identity = shortcut(x)

            x = conv1(x)
            x = norm1(x)
            x = nnx.gelu(x)

            x = conv2(x)
            x = norm2(x)
            x = nnx.gelu(x)

            return x + identity

        return forward

    def _pos_encoding(self, t: jax.Array, dim: int) -> jax.Array:
        """Applies sinusoidal positional encoding for time embedding.

        Args:
            t (jax.Array): The time embedding, representing the timestep.
            dim (int): The dimension of the output positional encoding.

        Returns:
            jax.Array: The sinusoidal positional embedding per timestep.

        """
        # Calculate half the embedding dimension.
        half_dim = dim // 2
        # Compute the logarithmic scaling factor for sinusoidal frequencies.
        emb = jnp.log(10000.0) / (half_dim - 1)
        # Generate a range of sinusoidal frequencies.
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        # Create the positional encoding by multiplying time embeddings with.
        emb = t[:, None] * emb[None, :]
        # Concatenate sine and cosine components for richer representation.
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb

    def _downsample(self, x: jax.Array) -> jax.Array:
        """Downsamples the input feature map with max pooling."""
        return nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

    def _upsample(self, x: jax.Array, target_size: int) -> jax.Array:
        """Upsamples the input feature map using nearest neighbor interpolation."""
        return jax.image.resize(x,
                              (x.shape[0], target_size, target_size, x.shape[3]),
                              method='nearest')

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Perform the forward pass through the U-Net using time embeddings."""

        # Time embedding and projection.
        t_emb = self._pos_encoding(t, 128) # Sinusoidal positional encoding for time.
        t_emb = self.time_mlp_1(t_emb) # Project and activate the time embedding
        t_emb = nnx.gelu(t_emb) # Activation function: `flax.nnx.gelu` (GeLU).
        t_emb = self.time_mlp_2(t_emb)

        # Project time embeddings for each scale.
        # Project to the correct dimensions for each encoder block.
        t_emb1 = self.time_proj1(t_emb)[:, None, None, :]
        t_emb2 = self.time_proj2(t_emb)[:, None, None, :]
        t_emb3 = self.time_proj3(t_emb)[:, None, None, :]

        # The encoder path with time injection.
        d1 = self.down_conv1(x)
        t_emb1 = jnp.broadcast_to(t_emb1, d1.shape) # Broadcast the time embedding to match feature map shape.
        d1 = d1 + t_emb1 # Add the time embedding to the feature map.

        d2 = self.down_conv2(self._downsample(d1))
        t_emb2 = jnp.broadcast_to(t_emb2, d2.shape)
        d2 = d2 + t_emb2

        d3 = self.down_conv3(self._downsample(d2))
        t_emb3 = jnp.broadcast_to(t_emb3, d3.shape)
        d3 = d3 + t_emb3

        # The bridge.
        b = self._downsample(d3)
        b = self.bridge(b)

        # The decoder path with skip connections.
        u3 = self.up_conv3(jnp.concatenate([self._upsample(b, d3.shape[1]), d3], axis=-1))
        u2 = self.up_conv2(jnp.concatenate([self._upsample(u3, d2.shape[1]), d2], axis=-1))
        u1 = self.up_conv1(jnp.concatenate([self._upsample(u2, d1.shape[1]), d1], axis=-1))

        # Final layers.
        x = self.final_norm(u1)
        x = nnx.gelu(x)
        return self.final_conv(x)
    
class UNet2(nnx.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 features: int,
                 time_emb_dim: int = 128,
                 *,
                 rngs: nnx.Rngs):
        """
        Initialize the U-Net architecture with time embedding.
        """
        self.features = features

        # Time embedding layers for diffusion timestep conditioning.
        self.time_mlp_1 = nnx.Linear(in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs)

        # Time projection layers for different scales.
        self.time_proj1 = nnx.Linear(in_features=time_emb_dim, out_features=features, rngs=rngs)
        self.time_proj2 = nnx.Linear(in_features=time_emb_dim, out_features=features * 2, rngs=rngs)

        # The encoder path.
        self.down_conv1 = self._create_residual_block(in_channels, features, rngs)
        self.down_conv2 = self._create_residual_block(features, features * 2, rngs)

        # The bridge connecting the encoder and the decoder.
        self.bridge = self._create_residual_block(features * 2, features * 4, rngs)

        # Decoder path with skip connections.
        self.up_conv2 = self._create_residual_block(features * 6, features * 2, rngs)
        self.up_conv1 = self._create_residual_block(features * 3, features, rngs)

        # Output layers.
        self.final_norm = nnx.LayerNorm(features, rngs=rngs)
        self.final_conv = nnx.Conv(in_features=features,
                                 out_features=out_channels,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding=((1, 1), (1, 1)),
                                 rngs=rngs)

    def _create_residual_block(self,
                              in_channels: int,
                              out_channels: int,
                              rngs: nnx.Rngs) -> Callable:
        """Creates a residual block with two convolutions and normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX PRNG keys.

        Returns:
            Callable: A function that represents the forward pass through the residual block.
        """

        # Convolutional layers with layer normalization.
        conv1 = nnx.Conv(in_features=in_channels,
                        out_features=out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=((1, 1), (1, 1)),
                        rngs=rngs)
        norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        conv2 = nnx.Conv(in_features=out_channels,
                        out_features=out_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=((1, 1), (1, 1)),
                        rngs=rngs)
        norm2 = nnx.LayerNorm(out_channels, rngs=rngs)

        # Projection shortcut if dimensions change.
        shortcut = nnx.Conv(in_features=in_channels,
                            out_features=out_channels,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            rngs=rngs)

        # The forward pass through the residual block.
        def forward(x: jax.Array) -> jax.Array:
            identity = shortcut(x)

            x = conv1(x)
            x = norm1(x)
            x = nnx.gelu(x)

            x = conv2(x)
            x = norm2(x)
            x = nnx.gelu(x)

            return x + identity

        return forward

    def _pos_encoding(self, t: jax.Array, dim: int) -> jax.Array:
        """Applies sinusoidal positional encoding for time embedding.

        Args:
            t (jax.Array): The time embedding, representing the timestep.
            dim (int): The dimension of the output positional encoding.

        Returns:
            jax.Array: The sinusoidal positional embedding per timestep.

        """
        # Calculate half the embedding dimension.
        half_dim = dim // 2
        # Compute the logarithmic scaling factor for sinusoidal frequencies.
        emb = jnp.log(10000.0) / (half_dim - 1)
        # Generate a range of sinusoidal frequencies.
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        # Create the positional encoding by multiplying time embeddings with.
        emb = t[:, None] * emb[None, :]
        # Concatenate sine and cosine components for richer representation.
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb

    def _downsample(self, x: jax.Array) -> jax.Array:
        """Downsamples the input feature map with max pooling."""
        return nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

    def _upsample(self, x: jax.Array, target_size: int) -> jax.Array:
        """Upsamples the input feature map using nearest neighbor interpolation."""
        return jax.image.resize(x,
                              (x.shape[0], target_size, target_size, x.shape[3]),
                              method='nearest')

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Perform the forward pass through the U-Net using time embeddings."""

        # Time embedding and projection.
        t_emb = self._pos_encoding(t, 128) # Sinusoidal positional encoding for time.
        t_emb = self.time_mlp_1(t_emb) # Project and activate the time embedding
        t_emb = nnx.gelu(t_emb) # Activation function: `flax.nnx.gelu` (GeLU).

        # Project time embeddings for each scale.
        # Project to the correct dimensions for each encoder block.
        t_emb1 = self.time_proj1(t_emb)[:, None, None, :]
        t_emb2 = self.time_proj2(t_emb)[:, None, None, :]

        # The encoder path with time injection.
        d1 = self.down_conv1(x)
        t_emb1 = jnp.broadcast_to(t_emb1, d1.shape) # Broadcast the time embedding to match feature map shape.
        d1 = d1 + t_emb1 # Add the time embedding to the feature map.

        d2 = self.down_conv2(self._downsample(d1))
        t_emb2 = jnp.broadcast_to(t_emb2, d2.shape)
        d2 = d2 + t_emb2

        # The bridge.
        b = self._downsample(d2)
        b = self.bridge(b)

        # The decoder path with skip connections.
        u2 = self.up_conv2(jnp.concatenate([self._upsample(b, d2.shape[1]), d2], axis=-1))
        u1 = self.up_conv1(jnp.concatenate([self._upsample(u2, d1.shape[1]), d1], axis=-1))

        # Final layers.
        x = self.final_norm(u1)
        x = nnx.gelu(x)
        return self.final_conv(x)


class DiffusionModel:
    def __init__(self,
                 model: UNet,
                 num_steps: int,
                 beta_min: float,
                 beta_max: float):

        self.model = model
        self.num_steps = num_steps
        self._time_schedule = self.time_schedule(num_steps)
        self._noise_schedule = self.subvp_noise_schedule(self._time_schedule, beta_min, beta_max)
        self._drift_schedule = self.subvp_drift_schedule(self._time_schedule, beta_min, beta_max)

    def time_schedule(self, num_steps: int) -> jnp.array:
        return jnp.linspace(0, num_steps, num_steps + 1) / num_steps
    
    @staticmethod
    def subvp_drift_schedule(t: jnp.array, beta_min: float = 0.1, beta_max: float = 20.0) -> jnp.array:
        return jnp.exp(-0.25*(t**2)*(beta_max - beta_min) - 0.5*t*beta_min)

    @staticmethod
    def subvp_noise_schedule(t: jnp.array, beta_min: float = 0.1, beta_max: float = 20.0) -> jnp.array:
        return 1.0 - jnp.exp(-0.5*(t**2)*(beta_max - beta_min) - t*beta_min)
    
    def forward(self, x: jax.Array, t: jax.Array, key: jax.Array) -> jax.Array:
        
        noise = jax.random.normal(key, x.shape)
        noisy_x = self._drift_schedule[t] * x + self._noise_schedule[t] * noise
        return noisy_x, noise
    
    def backward(self, x: jax.Array, key: jax.Array) -> jax.Array:

        x_t = x

        for t in reversed(range(self.num_steps)):

            delta_t = self._time_schedule[t] - self._time_schedule[t-1]

            t_batch = jnp.array([t] * x.shape[0])
            predicted_score = self.model(x_t, t_batch)

            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, x_t.shape) if t > 0 else 0

            x_t = x_t - delta_t*(self._drift_schedule[t]*x_t+ self._noise_schedule[t]**2 * predicted_score)  + jnp.sqrt(delta_t)*noise

        return x_t
    

if __name__ == "__main__":
   
    mnist = load_mnist()


    model = UNet2(in_channels=1, out_channels=1, features=4, rngs=nnx.Rngs(0))
    diffusion_model = DiffusionModel(model=model, num_steps=1, beta_min=0.1, beta_max=20.0)

    example = jnp.ones((1, 28, 28, 1))
    # example = jnp.array(mnist[0][5].reshape(1, 28, 28, 1))
    model(x=example, t=jnp.array([10]))
    # noisy_example = diffusion_model.forward(example, jnp.array([10]), jax.random.PRNGKey(0))
    # predicted = diffusion_model.backward(example, jax.random.PRNGKey(0))



