# Pre-Flight Checklist — Run This Before Any Training Job

Purpose: catch bugs like the sampler-math error we actually had (healthy loss,
broken samples) in minutes, not after a 22-hour run. Applies to all 4 projects
in this course. **Budget ~30–35 minutes on a T4** (smoke ~7, tiny-overfit ~12,
scheduler ~1, config ~1, throughput/resume ~15 — most of it unavoidable sample
grids). Cheaper than one wasted GPU day, every time.

Do not skip to "just run it" because the code looks right. The whole point of
this checklist is that code that looks right (correct-looking formula,
sensible-sounding config) was exactly what broke last time.

**In this repo, checks 1–6 are automated as notebook cells** in
`colab_train_t4.ipynb`, section "7. Pre-flight checklist" (between the smoke
test and the training cell). Run them top to bottom; each cell prints its own
PASS/FAIL/SKIP. A shared **`CFG_FLAGS` cell (0a) is the single source of
truth** for the T4 run's flags: cell 1c validates exactly that config, cell 1d
projects its hours from it, and the train cell (section 8) runs it — change
training behavior in 0a only. Locally, you can run the same logic with
`python preflight.py`-style snippets or by executing the notebook cells in any
GPU environment.

---

## 1. Tiny-overfit test (do this first, always — highest signal per minute spent)

- [ ] Take **8–16 samples** from your real dataset (not synthetic/random data —
      use the real thing so real data-loading and preprocessing code paths are
      exercised).
- [ ] Train until the loss genuinely bottoms out (early-stop at ε-MSE < 0.02 —
      ~0.08 is NOT converged for memorization purposes; watch the curve flatten).
- [ ] Sample from the model and check it can **near-perfectly reproduce these
      specific examples**.
- [ ] **Pass condition**: recognizable, close-to-input outputs (reconNCC(t=500)
      > 0.8, sample NCC > 0.5). **Fail condition**: static, blank, or
      unstructured noise even after loss has clearly converged.
- [ ] If it fails: the bug is structural (sampler, architecture, data
      pipeline) — do not proceed to a real run. This single test would have
      caught the `coef_xt` sampler bug in ~5 minutes instead of after 5,400
      steps of a real run.

**In this repo**: notebook cell "1a — tiny-overfit". Two stages, because a single
end-to-end score can false-alarm:
  * **Stage A — memorization probe**: 8 real sprites, overfit with early-stop at
    loss < 0.02 (cap 1500 steps), then ONE forward pass at fixed t ∈ {300, 500}:
    recover x0 from x_t and score NCC vs the original. No 1000-step error
    compounding. Untrained ≈ 0.2–0.5; overfit > 0.9. PASS bar: 0.8.
    ⚠️ Do NOT probe at low t (≤150): x_t is mostly signal there, so even an
    untrained model scores ~0.99 — the metric is only meaningful at t ≥ 300.
  * **Stage B — end-to-end sampler**: full 1000-step DDPM sample from pure
    noise; each sample is scored against its NEAREST of the 8 targets
    (best-match NCC). PASS bar: mean best-match > 0.5. Two rules baked in
    blood: (1) sampling does NOT preserve slot order — sample i is SOME
    training image, so positional pairing scores ~0.1 on a perfectly working
    sampler (this false-alarmed once); (2) residual training error is
    amplified 1000×, so ε-MSE ≈ 0.08 after only 300 overfit steps also scores
    ~0.1 despite a healthy pipeline — hence Stage A gates Stage B.
Failure routing: Stage A fails while loss still falls → UNDERTRAINED, raise
MAX_STEPS. Stage A fails with flat loss → STRUCTURAL BUG, stop. Stage B fails
after Stage A passes → SAMPLER BUG, stop.

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
  3. **Algebra identities** (schedule-agnostic; these catch sign/coefficient
     bugs that printed tables cannot): (a) the posterior-mean coefficient form
     vs its closed form at t ∈ {1, 100, 500, 999} (err < 1e-4); (b) a
     **`predict_x0` round-trip** for BOTH parameterizations — feed the true
     ε (or true v) and x0 must come back (err < 1e-5 at t ≤ 900). `predict_x0`
     is the exact function that held the original sampler bug. ⚠️ The check
     deliberately excludes t > 950: with `√ᾱ₉₉₉ ≈ 5e-5`, float32 rounding is
     amplified ~5000× and even a correct implementation shows ~2e-3 error —
     a naive round-trip at t=999 false-fails on precision alone.
  4. **Spot table** of the exact sampler coefficients at t ∈ {1, 100, 500, 999}.
  5. **Visual confirmation** (this is the one that catches what numbers don't):
     (a) matplotlib curves of `βₜ`, `√ᾱₜ`, `√(1−ᾱₜ)`, posterior variance over
     all 1000 steps; (b) a **forward-noise strip** — one real dataset image
     noised by `q_sample` at t = 0…999 with the SAME ε at every t (so the only
     thing changing is the schedule), showing the image intact until ~t≈400
     and pure noise by ~t≈800 with the cosine schedule.

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
cell "1c — config sanity" validates the `CFG_FLAGS` cell (the exact flags the
train cell uses — cell-defaults and train-flags drifting apart is its own
classic failure), plus an explicit demonstration of the exact-match trap, and
asserts the built model's parameter count sits in a 14–16M sanity band around
the known ~14.98M (catches accidental width/depth drift). Know the default
here: image 128 with mults (1,2,4) gives feature sizes 128/64/32, so
`attention_resolutions=(16, 8)` means **bottleneck-only attention** — 3
SelfAttention modules, not one per level.

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
- [ ] **Decode-check the ENTIRE dataset once** before a long run. A smoke run
      touches only a fraction of the files; one corrupt image mid-dataset
      otherwise kills the real run hours in. Iterating a DataLoader over all
      images through the real preprocessing path is minutes of CPU.

**In this repo**: notebook cell "1d — throughput & ETA" does all of it: (a) a
full-dataset decode sweep through the real `PixelArtDataset` path (toggle
`RUN_DECODE_SWEEP`); (b) s/step from the **median of per-row elapsed deltas**
in the isolated `logs/smoke_log.csv` — never total-elapsed ÷ steps, which the
smoke run's ~6-min sample grid inflates ~16×, and immune to smoke-cell re-runs
appending rows; (c) nvidia-smi sampled in a background thread DURING the
resume runs (median util, warn < 50%) instead of one idle snapshot; (d) total
budget projection including grid overhead at the `CFG_FLAGS` cadence. The T4
lesson: grids at the default 200-step cadence cost more than training does.

## 6. Resume/checkpoint integrity check

- [ ] Actually test `--resume_from` on a checkpoint before relying on it for a
      long, interruption-prone run (free-tier cloud sessions *will*
      disconnect).
- [ ] After resuming, confirm the loss curve continues smoothly from where it
      left off — a spike right after resume is a sign optimizer state (Adam
      momentum, etc.) wasn't restored correctly, not just cosmetic.

**In this repo**: notebook cell "1d" also runs a 40-step train → resume →
2-step continuation on an ephemeral checkpoint dir and compares the MEAN of
the last 3 pre-resume losses vs the mean of the 2 post-resume losses (2× =
WARN) — single-step losses swing ±50%, so a single-pair comparison
false-warns. GPU utilization is sampled during those same runs. Each run
emits a forced sample grid at its first step (by design) — that's your
item-4 undertrained grid. History note: the EMA-washout bug (legacy
checkpoint + warmup restart) is exactly the kind of thing this catches —
resumed loss looked fine while resumed samples regressed.

## 7. Only then: kick off the real run

- [ ] All of the above pass.
- [ ] You know your effective batch size, roughly how many epochs the run
      represents, and roughly how many hours it should take (rough s/step ×
      total steps).
- [ ] You know what "it's working" will look like at the 25%, 50%, and 100%
      marks — decide this *before* starting, so you're not guessing at 2am
      whether a half-finished run looks on track.

**In this repo, 20k steps, T4**: effective batch 64 (32×2), 50k images → ~26
epochs. Working looks like: 25% (5k) — coherent color fields and
rough character-shaped blobs; 50% (10k) — recognizable 4-view sprite layout,
white background dominates; 100% (20k) — clean LPC-style sprites with correct
silhouettes, some mode collapse toward common palettes is normal.

---

### The one-sentence version, if you only remember one thing

**Loss going down proves the training loop works — it proves nothing about the
sampler, the config, or anything downstream — so always check a second,
independent, cheap signal (tiny-overfit samples, a printed intermediate value,
GPU utilization) before trusting a long run to tell you the truth.**
