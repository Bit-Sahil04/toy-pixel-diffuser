"""DDPM forward process, noise schedule, training loss, and ancestral sampling.

This module owns everything timestep-related so the training loop and the
sampling loop can never drift out of sync (e.g. different normalizations or
different x0-reconstruction math). Model I/O is [-1, 1] (DDPM convention,
matching data.normalize()); sampling returns [0, 1] via data.denormalize().

Prediction target: "epsilon" is implemented; "v_prediction" is a wired seam
(same code path, different target and x0 reconstruction) — flip
config.prediction_target to try it if epsilon training is unstable.
"""

import math

import torch
import torch.nn.functional as F

from data import denormalize  # single source of truth for [-1,1] <-> [0,1]


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021). Better for small images."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999).float()


class Diffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        prediction_target: str = "epsilon",
        device: str | torch.device = "cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.prediction_target = prediction_target

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule: {schedule}")

        alphas = 1.0 - betas
        self.betas = betas.to(self.device)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)

    # ---------------------------------------------------------------- forward
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(ac_t)*x0 + sqrt(1-ac_t)*eps."""
        ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return ac * x0 + one_minus * noise

    # ------------------------------------------------------------------ loss
    def training_losses(self, model: torch.nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Sample t and noise, return MSE between prediction and target.

        epsilon-prediction:   target = eps                        (default)
        v-prediction:         target = ac*eps - sqrt(1-ac)*x0     (seam)
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        pred = model(x_t, t)
        if self.prediction_target == "epsilon":
            target = noise
        elif self.prediction_target == "v_prediction":
            ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
            one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            target = ac * noise - one_minus * x0
        else:
            raise ValueError(self.prediction_target)
        return F.mse_loss(pred, target)

    # -------------------------------------------------------------- sampling
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        """Reconstruct x0 from the model's prediction at timestep t."""
        ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        if self.prediction_target == "epsilon":
            return (x_t - one_minus * model_out) / ac.clamp_min(1e-8)
        # v-prediction: x_t = ac*x0 + sqrt(1-ac)*eps, v = ac*eps - sqrt(1-ac)*x0
        # => x0 = ac*x_t - sqrt(1-ac)*v
        return ac * x_t - one_minus * model_out

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        n: int,
        image_size: int,
        seed: int | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Ancestral (DDPM) sampling. Returns images in [0, 1], shape (n,3,S,S).

        Pass a FIXED seed across calls to denoise the same initial noise at
        different training stages — that's what makes the sample grids
        visually comparable over time (see train.py / config.sample_seed).
        """
        # Deterministic conv kernels: sampling is a chaotic 1000-step process,
        # so even 1e-6 kernel nondeterminism compounds into visibly different
        # grids, defeating the fixed-seed comparison. Scope: convs only (the
        # dominant op); this does not affect training determinism.
        cudnn_was = torch.backends.cudnn.deterministic
        bench_was = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            return self._sample(model, n, image_size, seed, verbose)
        finally:
            torch.backends.cudnn.deterministic = cudnn_was
            torch.backends.cudnn.benchmark = bench_was

    def _sample(self, model, n, image_size, seed, verbose) -> torch.Tensor:
        was_training = model.training
        model.eval()
        device = self.device
        # All randomness (initial noise AND per-step ancestral noise) comes
        # from this one seeded CPU generator, then moves to device — using the
        # global CUDA RNG here would break fixed-seed reproducibility.
        gen = torch.Generator(device="cpu").manual_seed(seed if seed is not None else 0)
        x = torch.randn(n, 3, image_size, image_size, generator=gen).to(device)

        steps = list(reversed(range(self.timesteps)))
        if verbose:
            from tqdm import tqdm
            steps = tqdm(steps, desc="sampling")
        for i in steps:
            t = torch.full((n,), i, device=device, dtype=torch.long)
            model_out = model(x, t)
            x0 = self.predict_x0(x, t, model_out).clamp(-1.0, 1.0)

            if i == 0:
                x = x0
                break
            alpha_cum = self.alphas_cumprod[i]
            alpha_cum_prev = self.alphas_cumprod[i - 1]
            beta = 1.0 - alpha_cum
            # posterior mean coefficients (standard DDPM ancestral step)
            coef_x0 = (alpha_cum_prev.sqrt() * beta) / (1.0 - alpha_cum)
            coef_xt = ((1.0 - alpha_cum_prev).sqrt() * beta) / (1.0 - alpha_cum)
            mean = coef_x0 * x0 + coef_xt * x
            var = (1.0 - alpha_cum_prev) / (1.0 - alpha_cum) * beta
            noise = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(device)
            x = mean + var.sqrt() * noise
        model.train(was_training)
        return denormalize(x)
