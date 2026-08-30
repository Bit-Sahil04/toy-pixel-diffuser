# Pre-Flight Checklist — Run This Before Any Training Job

Purpose: catch bugs like the sampler-math error we actually had (healthy loss,
broken samples) in minutes, not after a 22-hour run. Applies to all 4 projects
in this course. Takes 10–20 minutes total — cheaper than one wasted GPU day,
every time.

Do not skip to "just run it" because the code looks right. The whole point of
this checklist is that code that looks right (correct-looking formula,
sensible-sounding config) was exactly what broke last time.

**In this repo, checks 1–6 are automated as notebook cells** in
`colab_train_t4.ipynb`, section "7. Pre-flight checklist" (between the smoke
test and the training cell). Run them top to bottom; each cell prints its own
PASS/FAIL/SKIP. Locally, you can run the same logic with
`python preflight.py`-style snippets or by executing the notebook cells in any
GPU environment.

---

## 1. Tiny-overfit test (do this first, always — highest signal per minute spent)

- [ ] Take **8–16 samples** from your real dataset (not synthetic/random data —
      use the real thing so real data-loading and preprocessing code paths are
      exercised).
- [ ] Train for a few hundred steps (enough for loss to bottom out on this tiny
      set — watch the loss curve flatten).
- [ ] Sample from the model and check it can **near-perfectly reproduce these
      specific examples**.
- [ ] **Pass condition**: recognizable, close-to-input outputs. **Fail
      condition**: static, blank, or unstructured noise even after loss has
      clearly converged.
- [ ] If it fails: the bug is structural (sampler, architecture, data
      pipeline) — do not proceed to a real run. This single test would have
      caught the `coef_xt` sampler bug in ~5 minutes instead of after 5,400
      steps of a real run.

**In this repo**: notebook cell "1a — tiny-overfit". 16 real LPC sprites,
300 steps, then a full 1000-step DDPM sample. Automated metric: normalized
cross-correlation (NCC) between each output and its target; PASS if mean
NCC > 0.6 (an overfit this hard should reach >0.9). Side-by-side display:
originals on top, samples below — eyeball them too.

## 2. Numerically sanity-check anything you reimplemented from a formula

- [ ] For any math you translated from a paper/equation into code (noise
      schedules, loss weighting, sampler coefficients), **pick 2–3 concrete
      timesteps** and manually compute what the value *should* be.
- [ ] Print/log the actual value your code produces at those same timesteps.
- [ ] Compare side by side. A believable-looking number is not the same as a
      correct one — you're checking actual agreement, not vibes.
- [ ] If you didn't derive the formula yourself and are unsure, **diff your
      implementation against a trusted reference** (e.g. Hugging Face
      `diffusers`' `DDPMScheduler`, or the original paper authors' repo)
      rather than re-deriving it from scratch. Comparison is a completely
      legitimate way to verify correctness — you don't need to be able to
      prove it, just check it against something known-correct.

**In this repo**: notebook cell "1b — scheduler check". Does three things:
  1. **Table** at t ∈ {1, 100, 500, 999}: `βₜ`, `ᾱₜ`, `ᾱₜ₋₁`, posterior
     coefficients `coef_x0`/`coef_xt`, posterior variance — computed by
     `diffusion.py` next to values recomputed inline from the cosine closed
     form (Nichol & Dhariwal 2021) and the DDPM posterior formulas.
  2. **Cross-check vs `diffusers.DDPMScheduler`** (cosine) — allclose on the
     full betas vector. SKIPs with a note if diffusers isn't installed.
  3. **Visual confirmation** (this is the one that catches what numbers don't):
     (a) matplotlib curves of `βₜ`, `√ᾱₜ`, `√(1−ᾱₜ)`, posterior variance over
     all 1000 steps; (b) a **forward-noise strip** — one real dataset image
     noised by `q_sample` at t = 0…999, so you can SEE the schedule destroy
     the image at the right rate (image should stay mostly intact until
     ~t≈400 and be pure noise by ~t≈800 with the cosine schedule).

## 3. Config sanity pass — look for "silently does nothing" bugs

- [ ] For every conditional that depends on a config value (feature-flag
      checks, resolution/size checks, `in` membership checks), **actually
      compute what values it sees at runtime** and confirm the branch you
      expect is the one that fires.
- [ ] Specifically distrust exact-match checks (`==`, `in (a, b)`) on
      computed/derived values (like feature-map resolutions) — these are the
      classic place a config "looks right" but silently never activates, like
      `attention_resolutions=(16, 8)` never matching real feature maps of
      `[128, 64, 32]`.
- [ ] Print a one-line summary at startup of every major architectural choice
      actually taking effect (attention layers built: Y/N and where, prediction
      target used, EMA decay, effective batch size) — don't rely on reading the
      config file and trusting it matches runtime behavior.

**In this repo**: `train.py` prints a `[config-check]` block at startup of
every run (attention levels that actually got attention modules, prediction
target, EMA decay, effective batch, sampling/checkpoint cadence). Notebook
cell "1c — config sanity" does the same plus an explicit demonstration of the
exact-match trap. Know the default here: image 128 with mults (1,2,4) gives
feature sizes 128/64/32, so `attention_resolutions=(16, 8)` means
**bottleneck-only attention** — 3 SelfAttention modules, not one per level.

## 4. Two independent signals, not one

- [ ] Confirm you have **at least two signals that could disagree** with each
      other: training loss AND qualitative samples (or loss AND a held-out
      metric). Never run on loss alone.
- [ ] Before a long run, generate one sample grid from the current (even
      undertrained) checkpoint and actually look at it. Confirm the sampling
      code path runs end-to-end and produces *something* structured, even if
      low quality.
- [ ] Remember: a healthy loss curve validates the training loop only. It says
      nothing about sampling/inference code, because that's a separate code
      path.

**In this repo**: the tiny-overfit cell (loss + samples = two signals on one
small run) plus the resume-integrity cell below, which produces a grid from a
genuinely undertrained model — expect blurry blobs/uniform colors, NOT noise.
Static noise at healthy loss = sampler bug; stop.

## 5. Resource/throughput check (before committing GPU-hours)

- [ ] Run `nvidia-smi` (or equivalent) during a short smoke test. GPU
      utilization should be consistently high (~90%+). If it's low, you're
      I/O- or CPU-bound, not compute-bound — find the bottleneck
      (network-mounted disk logging/checkpointing, dataloader `num_workers`,
      per-step syncs like unnecessary `.item()` calls) before spending real
      training time on it.
- [ ] Check where logs and checkpoints are being written. Writing every step
      to a network mount (Google Drive, network disk) is a common, easy-to-miss
      slowdown — log locally, sync periodically instead. (This repo buffers
      CSV rows and flushes every 50 steps; checkpoints serialize locally once
      and are copied to Drive — don't regress this.)
- [ ] Confirm effective batch size (`batch_size × grad_accum_steps`) is
      actually what you intend, and sanity-check whether you actually need
      gradient accumulation — if a straight larger batch fits in VRAM without
      it, accumulation just adds overhead for no benefit.

**In this repo**: notebook cell "1d — throughput & ETA" samples `nvidia-smi`,
recomputes s/step from the smoke run's CSV, and projects total wall-clock
hours (including grid overhead at your `--sample_every`). The T4 lesson:
grids at the default 200-step cadence cost more than training does.

## 6. Resume/checkpoint integrity check

- [ ] Actually test `--resume_from` on a checkpoint before relying on it for a
      long, interruption-prone run (free-tier cloud sessions *will*
      disconnect).
- [ ] After resuming, confirm the loss curve continues smoothly from where it
      left off — a spike right after resume is a sign optimizer state (Adam
      momentum, etc.) wasn't restored correctly, not just cosmetic.

**In this repo**: notebook cell "1d" also runs a 40-step train → resume →
10-step continuation on an ephemeral checkpoint dir and auto-checks the first
post-resume loss against the last pre-resume loss (>3× = WARN). The resume
run emits a sample grid at its first step (by design) — that's your item-4
undertrained grid. History note: the EMA-washout bug (legacy checkpoint +
warmup restart) is exactly the kind of thing this catches — resumed loss
looked fine while resumed samples regressed.

## 7. Only then: kick off the real run

- [ ] All of the above pass.
- [ ] You know your effective batch size, roughly how many epochs the run
      represents, and roughly how many hours it should take (rough s/step ×
      total steps).
- [ ] You know what "it's working" will look like at the 25%, 50%, and 100%
      marks — decide this *before* starting, so you're not guessing at 2am
      whether a half-finished run looks on track.

**In this repo, 20k steps, T4, defaults**: effective batch 64 (32×2), 50k
images → ~19 epochs. Working looks like: 25% (5k) — coherent color fields and
rough character-shaped blobs; 50% (10k) — recognizable 4-view sprite layout,
white background dominates; 100% (20k) — clean LPC-style sprites with correct
silhouettes, some mode collapse toward common palettes is normal.

---

### The one-sentence version, if you only remember one thing

**Loss going down proves the training loop works — it proves nothing about the
sampler, the config, or anything downstream — so always check a second,
independent, cheap signal (tiny-overfit samples, a printed intermediate value,
GPU utilization) before trusting a long run to tell you the truth.**
