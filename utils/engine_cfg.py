"""
Classifier-Free Guidance (CFG) enabled training and sampling engines.

This module extends the standard DDPM training and sampling with CFG support.
Key features:
- GaussianDiffusionTrainer: Standard trainer (same as original, CFG logic in training loop)
- DDPMSampler_CFG: CFG-enabled sampler with dual prediction and guidance interpolation
"""

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class GaussianDiffusionTrainer(nn.Module):
    """
    Standard Gaussian Diffusion Trainer.

    This trainer is identical to the original implementation.
    CFG-specific logic (conditional dropout) is handled in the training loop (tools_cfg.py).
    """
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, y_onehot):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t, y_onehot)

        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
        loss = torch.sum(loss)
        return loss


class DDPMSampler_CFG(nn.Module):
    """
    DDPM Sampler with Classifier-Free Guidance (CFG).

    This sampler extends the standard DDPM sampling with CFG, which:
    1. Performs dual prediction: conditional and unconditional
    2. Interpolates predictions using guidance_scale
    3. Enables controllable trade-off between sample quality and diversity

    CFG Formula:
        ε_guided = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
                 = (1 - w) * ε_uncond + w * ε_cond
        where w = guidance_scale

    Guidance Scale Values:
        - 0.0: Pure unconditional generation
        - 1.0: Pure conditional generation (standard DDPM)
        - 3.0-7.0: Typical CFG range (better quality, less diversity)
        - 10.0+: Very strong guidance (may cause artifacts)
    """
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t, y_onehot):
        """
        Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$ (standard conditional).
        """
        epsilon_theta = self.model(x_t, t, y_onehot)
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * epsilon_theta

        # var is a constant
        var = extract(self.posterior_variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def cal_mean_variance_cfg(self, x_t, t, y_cond, y_uncond, guidance_scale):
        """
        Calculate mean and variance with Classifier-Free Guidance.

        This method performs:
        1. Conditional noise prediction: ε_cond = model(x_t, t, y_cond)
        2. Unconditional noise prediction: ε_uncond = model(x_t, t, y_uncond)
        3. Guided prediction: ε_guided = ε_uncond + w * (ε_cond - ε_uncond)

        Parameters:
            x_t: Noisy input at timestep t
            t: Timestep tensor
            y_cond: Conditional class embedding (one-hot)
            y_uncond: Unconditional embedding (all zeros)
            guidance_scale: CFG guidance strength (w)

        Returns:
            mean, var: Mean and variance for sampling x_{t-1}
        """
        # Conditional prediction
        epsilon_cond = self.model(x_t, t, y_cond)

        # Unconditional prediction (y_uncond is all zeros)
        epsilon_uncond = self.model(x_t, t, y_uncond)

        # Apply classifier-free guidance interpolation
        epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

        # Calculate mean using guided epsilon
        mean = (extract(self.coeff_1, t, x_t.shape) * x_t -
                extract(self.coeff_2, t, x_t.shape) * epsilon_guided)

        # Variance remains the same
        var = extract(self.posterior_variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, y_cond, y_uncond, guidance_scale):
        """
        Sample one reverse diffusion step with CFG.

        Parameters:
            x_t: Current noisy sample
            time_step: Current timestep (scalar int)
            y_cond: Conditional embedding
            y_uncond: Unconditional embedding (all zeros)
            guidance_scale: CFG guidance strength

        Returns:
            x_{t-1}: Less noisy sample
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)

        # Select sampling strategy based on guidance_scale
        if guidance_scale == 0.0:
            # Pure unconditional generation
            mean, var = self.cal_mean_variance(x_t, t, y_uncond)
        elif guidance_scale == 1.0:
            # Pure conditional generation (standard DDPM)
            mean, var = self.cal_mean_variance(x_t, t, y_cond)
        else:
            # CFG sampling (guidance_scale > 1.0 typically)
            mean, var = self.cal_mean_variance_cfg(x_t, t, y_cond, y_uncond, guidance_scale)

        # Sample from Gaussian (or deterministic at t=0)
        z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        if torch.isnan(x_t_minus_one).int().sum() != 0:
            raise ValueError("nan in tensor!")

        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, y_cond, guidance_scale=3.0,
                only_return_x_0: bool = True, interval: int = 1):
        """
        CFG-enabled sampling forward pass.

        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            y_cond: Class-conditional one-hot tensor with shape (batch_size, num_classes).
            guidance_scale: CFG guidance strength.
                - 0.0: unconditional
                - 1.0: conditional (no guidance)
                - 3.0-7.0: typical CFG (recommended)
                - 10.0+: very strong guidance
            only_return_x_0: If True, only return the final result $x_0$.
            interval: Interval for saving intermediate pictures (only if only_return_x_0=False).

        Returns:
            if `only_return_x_0 = True`: tensor with shape (batch_size, channels, height, width)
            otherwise: tensor with shape (batch_size, sample, channels, height, width)
        """
        # Create unconditional embedding (all zeros)
        y_uncond = torch.zeros_like(y_cond)

        x = [x_t]
        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as time_steps:
            for time_step in time_steps:
                x_t = self.sample_one_step(x_t, time_step, y_cond, y_uncond, guidance_scale)

                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                time_steps.set_postfix(ordered_dict={
                    "step": time_step + 1,
                    "sample": len(x),
                    "cfg_scale": guidance_scale
                })

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
