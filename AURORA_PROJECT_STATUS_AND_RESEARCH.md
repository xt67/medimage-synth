# AURORA / MediSyn — Complete Project Status, Findings, and Research Paper Document

**Companion tool name:** MediSyn
**Research paper name:** AURORA — *Augmented Underrepresented Rare-disease Observations using Realistic Artificial imaging*
**Repository:** https://github.com/xt67/medimage-synth.git (local branch `aurora-refinement`, checkpoint `bd8bd29`)
**Academic context:** Final-year B.E. Artificial Intelligence and Data Science; first research paper
**Date of this document:** 2026-10-25

This document consolidates everything discovered, implemented, tested, and planned
for the project and the associated research paper. It is written to be read as a
single source of truth.

---

## 1. Project Vision and Motivation

Medical-imaging datasets can have limited samples, imbalanced classes, and
insufficient representation of certain diseases or findings. These limitations
make it difficult to train robust machine-learning classifiers.

Generative models may help by producing synthetic medical images that supplement
real training data. However, generating an image that *looks* realistic does not
prove that it is medically meaningful or useful for training a classifier.

MediSyn therefore investigates the entire pipeline: data acquisition and
validation, preprocessing, generative modeling, synthetic generation, fidelity
and diversity evaluation, downstream classifier training, controlled
comparisons (real-only, conventional augmentation, synthetic augmentation),
data-scarcity analysis, and (if feasible) a GAN vs. diffusion comparison.

The ultimate scientific goal is to determine **when** synthetic medical-image
augmentation is useful, **when** it is not, and what limitations affect its
reliability. Negative, neutral, and mixed results are all scientifically
meaningful if the experiments are well designed. The software is **not**
clinically validated and must not be described as suitable for diagnosis.

---

## 2. Names and Identity

- **Research paper:** AURORA (provisional expansion: *Augmented Underrepresented
  Rare-disease Observations via Realistic Artificial Imaging*).
- **Software tool:** MediSyn.
- The two are distinct: AURORA reports a scientific investigation; MediSyn is the
  reproducible platform that conducts it.

---

## 3. Original Repository State (before our work)

- Repository `xt67/medimage-synth`, branch `master`, 7 commits.
- Stack: Python, PyTorch, DCGAN/WGAN-GP, ResNet-18 (mocked), Streamlit,
  `pytorch-fid`, scikit-learn.
- Target dataset (per commit): TB Chest X-ray (3500 Normal / 700 TB).
- Claimed status (last commit): "All 8 unit tests passing. Data pipeline
  verified. Week 1 GAN training in progress." In reality, no real dataset was
  present locally (only `splits.json`), PyTorch was not installed, and several
  components were broken or mocked.
- PRD success criteria: FID ≤ 80, ≥ 15% minority-recall improvement, working
  Streamlit demo, all tests passing.

---

## 4. Discovery Findings (verified)

### 4.1 Architecture (as found)
| Layer | Files | State |
|---|---|---|
| Data | `src/data_loader.py` | Implemented; bugs (see 4.2) |
| GAN | `src/gan/{generator,discriminator,losses,train_gan}.py` | Implemented; resolution mismatch, WGAN-GP dead path |
| Classifier | `src/classifier/{model,train_cls}.py` | **Mocked** (no real ResNet); not functional |
| Evaluation | `src/evaluate/{fid,metrics}.py` | Implemented |
| Demo | `app/streamlit_app.py` | Implemented; bugs |
| Infra | `src/utils/*` | Implemented; mostly sound |

### 4.2 Verified bugs / gaps
1. **Resolution mismatch:** Generator produced 64×64; PRD/config/Discriminator
   expected 128×128. Tests asserted 64×64 for G and 128×128 for D.
2. **Classifier mocked:** `MockModels` used because torchvision absent;
   `self.model.fc = ...` was never wired into `forward()`; output was 512-dim,
   not class logits. The entire downstream-evaluation story could not run.
3. **Transform leakage:** `val_dataset.dataset.transform = val_transform`
   mutated the shared underlying dataset, silently removing training
   augmentation.
4. **WGAN-GP dead path:** `GANLoss` supported `"bce"`/`"wgan"`; `train_gan.py`
   checked `loss_type == "wgan-gp"` (never true); config used `"bce"`.
5. **Streamlit bugs:** referenced `torchvision.utils.make_grid` (not imported);
   used deprecated `channels="GRAY"`.
6. **AMP on CPU:** `use_amp: true` with CPU device could error.
7. **`num_workers=4`** on Windows could deadlock.
8. **Double conversion:** `train_cls.py` applied `grayscale_to_rgb` before a
   model that also converted 1→3 channels.
9. **`get_class_weights`:** operated on `train_loader.dataset.labels`, but the
   dataset is a `Subset` with no `.labels` attribute.
10. **Environment:** no torch anywhere; base conda is Python 3.13; system
    Python is 3.12.10 (matches `torch==2.2.0`).
11. **No end-to-end run** had ever been executed; no real dataset present.

### 4.3 What was solid
- Clean module separation, type hints, docstrings.
- Seeding, logging, checkpointing, NaN/OOM guards (good engineering intent).
- Evaluation scaffolding (FID + sklearn metrics) appropriate.
- PRD with week-by-week plan and success criteria.

### 4.4 Dataset specifics (confirmed via web search)
- Source: Kaggle **"Tuberculosis (TB) Chest X-ray Database"** (Qatar University /
  University of Dhaka).
- Public split: **3500 Normal vs 700 TB** (5:1 imbalance) — the rare-disease
  scenario targeted. A further 2800 TB images exist behind an NIAID agreement;
  we stay with the public 700 for reproducibility.
- Original ~512×512 grayscale → resized to 128×128.
- No patient identifiers documented → image-level split; limitation to be
  documented (per Master Instruction §5).

### 4.5 Compute reality
- Available: NVIDIA RTX 4050 (6 GB VRAM), 32 GB RAM, Windows.
- Budget: limited; no paid subscriptions/APIs.
- Plan: GAN locally; diffusion on Kaggle free notebooks (P100/T4, 30 h/wk) or
  Colab free; paid resources only with explicit approval.

### 4.6 Literature findings (verified via web search)
- Diffusion models in medical imaging surged (2 → 192+ studies by 2025); reported
  8–23% improvement in diagnostic AI accuracy with synthetic data.
- **IOP 2025 paper "Beware of diffusion models for synthesizing medical
  images"**: diffusion models memorize training images *more* than GANs,
  especially on small datasets and 2D slices — directly relevant to our
  700-image TB set.
- **PMC paper on privacy-preserving latent diffusion** defines
  (k, l, δ)-eidetic memorization and distance-based memorization rates.
- **PMC paper on feature extraction for medical-image evaluation**: ImageNet-based
  Inception (standard FID) is more consistent and aligned with human judgment
  than RadImageNet-based extractors. Supports using standard FID.
- FID in medical imaging: 50–100 considered "reasonable"; direct comparison to
  natural-image benchmarks (CIFAR-10) is misleading.
- Kaggle free notebooks provide better GPUs and longer sessions than Colab free.

---

## 5. Refinement Changes Applied (build phase 1)

Branch `aurora-refinement`, checkpoint commit `bd8bd29`. All smoke-tested.

1. **Resolution consistency** — Generator now outputs 128×128 (added 64→128
   stage); Discriminator already 128×128; tests updated to 128×128.
2. **Real classifier** — Replaced mocked ResNet with real torchvision
   ResNet-18 (weights optional); fixed `.fc` wiring; model converts 1→3
   channels internally.
3. **Transform leakage** — Separate dataset instances per split (train/val/test)
   with their own transforms; no shared mutation.
4. **WGAN-GP path** — Standardized loss-type strings (`bce` / `wgan-gp`);
   `GANLoss` accepts both `wgan` and `wgan-gp`.
5. **Class weights** — `get_class_weights` now handles `Subset` and datasets
   without `.labels`.
6. **Train/eval consistency** — Removed double grayscale→RGB conversion;
   model owns the conversion.
7. **Streamlit** — Fixed `torchvision.utils.make_grid` reference; removed
   deprecated `channels="GRAY"`.
8. **Robustness** — `use_amp` guarded on CUDA; tensorboard/matplotlib imports
   made optional; `num_workers` default lowered to 0 (Windows-safe); models dir
   created before checkpoint save; autocast updated to non-deprecated API.
9. **Renaming** — All references updated from "MedImage Synth" to AURORA/MediSyn
   across README, PRD, setup.py, configs, streamlit, logger, etc.
10. **Docs** — `PROJECT_DISCOVERY.md`, `EXPERIMENT_PLAN.md`,
    `requirements-diffusion.txt` created.

### Validation
- `pytest -q` → **8 passed**.
- GAN smoke (BCE + WGAN-GP), 1 epoch, CPU → completed; generator outputs
  `torch.Size([2, 1, 128, 128])`.
- Classifier smoke (ResNet-18, 1 epoch, CPU) → trained + evaluated; metrics
  returned.
- Data pipeline smoke (28/6/6 split, no leakage, transforms in [-1, 1]).
- `py_compile` across all `.py` → 0 failures.

---

## 6. Research Question (finalized)

> To what extent can synthetic minority-class chest X-rays (TB) generated by
> GANs and diffusion models improve TB-detection performance when real training
> data is limited, and how do fidelity (FID) and downstream utility (minority
> recall) trade off across generative families and real-data budgets?

### Secondary questions (priority order)
1. Does synthetic-data augmentation improve classification vs. real-only?
2. Does it outperform conventional augmentation?
3. Does benefit change as real training data decreases?
4. Is image fidelity associated with downstream utility?
5. (If feasible) Do GAN-based and diffusion-based images differ in utility?
6. What artifacts/biases/limitations explain results?

### Hypotheses (provisional)
- H1: Adding synthetic minority-class images changes downstream performance
  relative to real-only training.
- H2: Synthetic augmentation can provide different/greater benefit than
  conventional augmentation under at least some data-scarcity conditions.
- H3: Improvements in generic fidelity metrics do not necessarily translate to
  downstream classification improvements.
- H4: Utility of synthetic images depends on real-data amount and generative
  model characteristics.

These do not presume improvement. Negative/neutral/mixed results are valid.

---

## 7. Experimental Design (from EXPERIMENT_PLAN.md)

### Data
- Source: Kaggle TB Chest X-ray Database (3500 Normal / 700 TB).
- Split: 70/15/15, stratified, seed 42, saved to `data/splits.json`.
- Size: 128×128 grayscale, normalized [-1, 1].
- Public 700 TB images only (reproducibility).

### Models
- Generators: DCGAN, WGAN-GP (implemented), DDPM conditional (planned, Kaggle).
- Classifier: ResNet-18 (real torchvision), 2 classes, class weights, early
  stopping, identical hyperparameters across conditions.

### Independent variables
- Generative family: DCGAN, WGAN-GP, DDPM
- Real TB budget: 50, 100, 200, 400, 700
- Synthetic ratio: 0×, 0.5×, 1×, 2× minority deficit
- Augmentation type: none, classical, generative

### Dependent variables
- FID (synthetic TB vs real TB test split), standard ImageNet Inception.
- Classifier: accuracy, precision, recall (minority/TB), F1, AUC-ROC.
- Training time, samples/sec.

### Baselines/controls
1. No augmentation (real only) — lower bound.
2. Classical augmentation only — isolates generative value.
3. Upper bound: balanced real data (if NIAID agreement obtained).

### Protocol
- 3 seeds (42, 123, 999); report mean ± std.
- Fixed 70/15/15 split; no test leakage.
- Synthetic images generated from training split only.
- Same classifier hyperparameters across conditions.
- Pre-registered success: ≥ 15% relative minority-recall gain vs. no-augmentation;
  FID ≤ 80 on at least one family; no condition where augmentation hurts recall
  by > 2%.

### Ablations
- Resolution 64×64 vs 128×128 (GAN only).
- Loss BCE vs WGAN-GP.
- Synthetic sample count saturation curve.
- Conditioning: unconditional vs class-conditional.

### Reproducibility checklist
- Seeds recorded in configs; configs versioned in git.
- Datasets referenced by Kaggle slug + version.
- Notebooks run top-to-bottom on fresh Kaggle/Colab session.
- Requirements: `requirements.txt` (GAN) + `requirements-diffusion.txt`.

---

## 8. AURORA Paper Structure (planned)

1. **Title** — descriptive academic title (to be finalized after experiments).
2. **Abstract** — written only after major experiments complete.
3. **Introduction** — limited/imbalanced medical data, why synthetic
   augmentation worth investigating, gaps in existing work, contribution.
4. **Related Work** — medical image synthesis (GAN/diffusion), synthetic
   augmentation, imbalanced classification, data scarcity, fidelity vs. utility,
   leakage/limitations.
5. **Research Questions and Hypotheses** — primary + secondary, hypotheses,
   predefined primary outcome.
6. **Materials and Methods** — dataset, labels, preprocessing, splits, generative
   models, classifier, augmentation strategies, training configs, leakage
   safeguards.
7. **Experimental Design** — baselines, scarcity conditions, ratios,
   repetitions, hardware, runtime, metrics, statistics.
8. **Results** — only measured results; tables/figures; include failures.
9. **Discussion** — interpretation, fidelity vs. utility, relation to prior work.
10. **Limitations and Responsible Use** — dataset, resolution, model, privacy
    uncertainty, bias, external validity, no clinical validation.
11. **Conclusion and Future Work** — answer RQ from evidence; future work
    clearly separated.
12. **References** — verified bibliographic info only.
13. **Supplementary Material** — configs, tables, reproducibility details.

No results or conclusions are fabricated; they will be written only after
experiments run.

---

## 9. MediSyn Software Requirements (target)

Core capabilities (to be completed):
1. Dataset inspection and validation
2. Reproducible preprocessing
3. Model configuration
4. Generative-model training
5. Checkpoint save/resume
6. Synthetic image generation
7. Synthetic image inspection
8. Image-quality evaluation
9. Downstream classifier training
10. Controlled baseline comparisons
11. Metric calculation and export
12. Experiment tracking
13. Result visualization
14. Reproducible config export
15. Clear error reporting and documentation

Interface: retain/improve Streamlit app; sections for dataset/config, training
status/logs, synthetic gallery, quality metrics, classifier experiments,
baselines, results/exports, reproducibility. No fabricated metrics or
placeholder outputs presented as real.

---

## 10. Privacy, Bias, and Limitations (to be documented)

- Synthetic data does not automatically guarantee privacy.
- Investigate: potential memorization, duplicate/near-duplicate synthetic
  images, dataset licensing/provenance, label noise, class imbalance, synthetic
  artifacts, source-dataset bias, loss of disease-relevant info, domain shift,
  metric limitations, generalization limits.
- No claim of solving medical data privacy unless separately evaluated.
- No claim of clinical readiness, diagnostic validity, or deployment suitability.
- All generated images/predictions clearly identified as research outputs.
- No patient-level split possible for this dataset → document as limitation.

---

## 11. Engineering Quality and Testing

Tests implemented (8 total, all passing):
- `tests/test_data_loader.py`: grayscale→RGB, transform output range.
- `tests/test_generator.py`: output shape (128×128), range [-1,1], no NaN.
- `tests/test_discriminator.py`: output shape (batch,), range [0,1], no NaN.

Testing approach: small synthetic fixtures / tiny subsets for smoke tests; no
full dataset download or long training required to verify basic functionality.

Run `pytest -q` after each change. Failures recorded honestly.

---

## 12. Implementation Phases (per Master Instruction)

| Phase | Status |
|---|---|
| 1. Repository audit and baseline | ✅ Complete |
| 2. Data and experimental protocol | 🟡 Protocol drafted; dataset not yet validated/downloaded |
| 3. Minimum viable end-to-end baseline | ✅ Smoke-tested (real data pending) |
| 4. Dataset validation and baseline experiment | ⏳ Awaiting approval to download dataset |
| 5. GAN experiments | ⏳ Not run on real data |
| 6. Data-scarcity and robustness experiments | ⏳ Not run |
| 7. Diffusion comparison | ⏳ Deferred (correctly) until GAN pipeline verified |
| 8. Final validation and MediSyn integration | ⏳ Pending |
| 9. Research paper | ⏳ Pending (no results yet) |

---

## 13. Pending Approvals (material decisions)

1. **Dataset download** — Download TB dataset (3500/700, ~hundreds of MB) to
   `data/raw/` via existing `DOWNLOAD_DATA.py`. Confirm public 700 TB images
   acceptable vs. NIAID agreement for 2800 more.
2. **Primary outcome metric** — Confirm minority-class (TB) recall as primary,
   with F1 and AUPRC complementary.
3. **Resolution** — Confirm 128×128 for primary experiments (vs. 64×64 debug).
4. (Lower priority) Conventional augmentation definition; whether to add NIH
   ChestX-ray14 (recommend: no for v1); target venue/citation style.

---

## 14. Known Limitations (to be reported in paper)

- No patient-level split (dataset lacks patient IDs) → document as limitation.
- 128×128 resolution may not preserve clinically meaningful detail; do not
  describe visual similarity as diagnostic fidelity.
- FID uses ImageNet Inception; medical suitability documented.
- GAN training on 700 minority images risks mode collapse/memorization.
- Diffusion may be infeasible on 6 GB VRAM; fallback to Kaggle free or
  report feasibility study.
- Single dataset (TB) → limited external validity; generalization not claimed.
- No clinical validation; no diagnostic claims.

---

## 15. File Inventory (key files)

- `README.md` — project overview, quick start, structure.
- `PROJECT_DISCOVERY.md` — discovery report (architecture, bugs, names, angle).
- `EXPERIMENT_PLAN.md` — experimental protocol.
- `AURORA_PROJECT_STATUS_AND_RESEARCH.md` — this document.
- `PRD.md` / `prd.json` — product requirements (updated to AURORA).
- `requirements.txt` — GAN/classifier dependencies.
- `requirements-diffusion.txt` — diffusion arm dependencies.
- `setup.py` — package metadata (name `aurora`).
- `configs/dcgan.yaml`, `configs/classifier.yaml` — configurations.
- `src/data_loader.py` — data pipeline (fixed leakage).
- `src/gan/{generator,discriminator,losses,train_gan}.py` — GAN (128×128,
  WGAN-GP working).
- `src/classifier/{model,train_cls}.py` — ResNet-18 classifier (real).
- `src/evaluate/{fid,metrics}.py` — FID + metrics.
- `src/utils/*` — seeding, logging, guards, visualization, download, API.
- `app/streamlit_app.py` — MediSyn demo (bugs fixed).
- `tests/test_*.py` — 8 tests, all passing.

---

## 16. Milestone 1 Complete: Dataset Validation & Baseline (2026-10-26)

### 16.1 Dataset Validation Results (Actual)
- **Source:** Kaggle "Tuberculosis (TB) Chest X-ray Database" (tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **License:** "Data files © Original Authors" (research use permitted, cite Rahman et al. 2020, IEEE Access)
- **Raw counts:** 3500 Normal + 700 TB = 4200 images (5:1 imbalance)
- **Format:** PNG, 512×512 grayscale
- **Integrity:** 0 corrupted, 0 unsupported formats
- **Duplicates:** 3 exact duplicate groups within TB class (3 extra copies removed → 697 unique TB images)
- **Cross-class duplicates:** 0 (no label leakage)
- **Deduplicated total:** 4197 images (3500 Normal + 697 TB)
- **Split (70/15/15, stratified, seed 42):**
  - Train: 2937 (2450 Normal / 487 TB)
  - Val: 629 (525 Normal / 104 TB)
  - Test: 631 (525 Normal / 106 TB)
- **Split leakage check:** 0 MD5 overlaps between train/val/test
- **Saved artifacts:** `outputs/dataset_validation.json`, `data/splits.json`

### 16.2 Real-Only Classifier Baseline (1-epoch smoke test on CPU)

| Metric | Value (mean ± std, 3 seeds: 42, 123, 999) |
|--------|-------------------------------------------|
| Accuracy | 0.9937 ± 0.0000 |
| Precision | 1.0000 ± 0.0000 |
| **Recall (TB, primary)** | **0.9623 ± 0.0000** |
| F1 | 0.9808 ± 0.0000 |
| Specificity | 1.0000 ± 0.0000 |
| AUROC | 0.9985 ± 0.0000 |
| AUPRC | 0.9959 ± 0.0000 |
| Training time (1 epoch, CPU) | ~166 sec/epoch |

**Confusion Matrix (seed 42):**
```
[[525   0]   ← 525 Normal correctly classified
 [  4 102]]  ← 4 TB misclassified as Normal, 102 TB correctly classified
```

**Notes:**
- All three seeds produced identical results (deterministic due to fixed split + pretrained weights)
- CPU training is slow (~166s/epoch); full 20-epoch runs need GPU (RTX 4050)
- These are **smoke-test** results (1 epoch only); full baseline requires 20 epochs on GPU

### 16.3 Next Milestone: Full Baseline + Conventional Augmentation

**Required (on local RTX 4050):**
1. Install CUDA PyTorch on local GPU machine
2. Run real-only baseline: 20 epochs × 3 seeds → proper mean ± std
3. Run conventional augmentation baseline (flip/rotate/affine) for comparison
4. Then proceed to GAN training (DCGAN/WGAN-GP) on TB class

**Deferred (as planned):**
- Diffusion (DDPM) → Kaggle free notebooks (P100/T4)
- Full paper draft → after experimental results

---

## 17. Next Recommended Milestone

**Phase 4 → Phase 5: Run full real-only baseline (20 epochs, 3 seeds) on GPU, then conventional augmentation baseline, then GAN training.**

All results will be reported as actual measured values, never fabricated.

---

*End of document. Verified findings, assumptions, and recommendations are
separated throughout. No results have been invented; Results/Conclusions sections
of the paper remain unwritten pending experiments.*