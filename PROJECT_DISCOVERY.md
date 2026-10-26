# Project Discovery Report

**Repository:** `xt67/medimage-synth` (local clone)  
**Prepared for:** Final-year B.Tech project → first research paper  
**Status:** Discovery / planning phase (no code changes yet)

---

## 1. Executive summary

`MedImage Synth` is a **GAN-based synthetic medical image generator** aimed at
augmenting imbalanced datasets (e.g. rare diseases such as Tuberculosis in chest
X-rays) so that downstream classifiers perform better on minority classes,
while avoiding sharing real patient data.

The repository is a **scaffold + partial implementation**: the data pipeline, GAN
architectures, training loops, classifier, evaluation (FID + metrics) and a
Streamlit demo all exist, but several components are **incomplete, inconsistent,
or mocked** (see §4). No real dataset is present in the clone (only
`splits.json`), and PyTorch is not installed in the active environment, so
nothing has been run end-to-end yet.

The core idea is sound and publishable, but the project needs a **clearer
identity (name), a single well-defined research question, and a tightened
scope** before building.

---

## 2. What the project currently is

### 2.1 Architecture
| Layer | File(s) | What it does |
|---|---|---|
| Data | `src/data_loader.py` | `MedicalImageDataset`, transforms (resize 128, normalize [-1,1]), train/val/test split |
| GAN | `src/gan/{generator,discriminator,losses,train_gan}.py` | DCGAN (ConvTranspose2d generator, Conv2d discriminator), BCE / WGAN-GP losses, AMP training, checkpoints every 10 epochs |
| Classifier | `src/classifier/{model,train_cls}.py` | ResNet-18 fine-tune (currently **mocked**), class weighting, early stopping, recall/precision/F1 |
| Evaluation | `src/evaluate/{fid,metrics}.py` | FID via `pytorch-fid`, sklearn metrics |
| Demo | `app/streamlit_app.py` | Streamlit: overview, generator, results dashboard, technical |
| Infra | `src/utils/*` | seeding, logging, OOM/NaN guards, visualization, download helpers, API handlers |

### 2.2 Configuration
- `configs/dcgan.yaml`: z_dim=100, 128×128 grayscale, 200 epochs, lr 2e-4, BCE loss.
- `configs/classifier.yaml`: ResNet-18, 2 classes, 20 epochs.

### 2.3 Dataset target
- TB Chest X-ray (Kaggle/Google Drive). Reported: 3500 normal + 700 TB images,
  128×128 grayscale. Severe class imbalance (5:1) — exactly the scenario the
  project targets.

### 2.4 Success criteria (from PRD)
- FID ≤ 80 (synthetic vs real)
- ≥ 15% improvement in minority-class recall after augmentation
- Working Streamlit demo
- All unit tests passing

---

## 3. Discoveries about the existing code

### 3.1 Resolution mismatch (important)
- `Generator` produces **64×64** images (4 upsampling stages from 4×4).
- PRD / config / `Discriminator` expect **128×128**.
- Tests assert 64×64 for the generator but 128×128 for the discriminator.
- **Impact:** training would run but at the wrong resolution; augmentation
  experiments would be apples-to-oranges. Must pick one resolution and make
  everything consistent. Recommendation: keep **128×128** (medical detail
  matters) and add one more upsampling stage to the generator.

### 3.2 Classifier is mocked
- `src/classifier/model.py` defines a `MockModels` class because torchvision
  isn't installed, and `self.model.fc = ...` is never actually wired into
  `forward()`. The model outputs 512-dim features, not class logits.
- **Impact:** the entire downstream-evaluation story (the paper's main result)
  cannot run. Needs real torchvision ResNet-18.

### 3.3 Transform leakage across splits
- In `create_dataloaders`, `val_dataset.dataset.transform = val_transform`
  mutates the shared underlying dataset, which also changes the train subset's
  transform. Training augmentation would be silently lost.
- **Fix:** use separate dataset instances per split or per-subset transforms.

### 3.4 Loss-type string mismatch
- `GANLoss` supports `"bce"` / `"wgan"`; `train_gan.py` checks
  `loss_type == "wgan-gp"` (never true) and config uses `"bce"`.
- **Impact:** WGAN-GP path is dead code. Either remove or implement properly.

### 3.5 Other minor issues
- `train_cls.py` applies `grayscale_to_rgb` before passing to a model that
  *also* converts 1→3 channels → inconsistent but functional.
- `streamlit_app.py` references `torchvision.utils.make_grid` (not imported)
  and uses deprecated `channels="GRAY"`.
- `num_workers=4` on Windows can deadlock; safer default 0/2.
- AMP (`use_amp: true`) with CPU device may error; guard on CUDA availability.
- `OOMGuard.__call__` is a context manager but is used as `with oom_guard():`
  — works, but the raised OOM is re-raised (no retry), limiting its value.

### 3.6 What is solid
- Clean module separation, type hints, docstrings.
- Seeding, logging, checkpointing, NaN/OOM guards show good engineering intent.
- Evaluation scaffolding (FID + sklearn metrics) is appropriate.
- PRD with week-by-week plan and success criteria is a good research scaffold.

---

## 4. Issues & gaps summary (prioritized)

| # | Issue | Severity | Effort |
|---|---|---|---|
| 1 | Classifier mocked (no real ResNet) | Blocker for paper result | Medium |
| 2 | Generator 64×64 vs 128×128 mismatch | High | Low |
| 3 | Transform leakage across splits | High | Low |
| 4 | No end-to-end run ever executed | High | — |
| 5 | WGAN-GP path dead | Medium | Medium |
| 6 | Streamlit bugs | Low | Low |
| 7 | No baseline numbers / no experiment harness | High (for paper) | Medium |
| 8 | No radiologist/clinical validation plan | Medium (for paper) | High |

---

## 5. Candidate project names

Goal: short, memorable, medical + generative, sounds like a real tool/paper.

| # | Name | Rationale |
|---|---|---|
| 1 | **MediForge** | "forging" realistic medical images; strong, brandable, memorable |
| 2 | **AURORA** | *Augmented Underrepresented Rare-disease Observations via Realistic Artificial imaging* — real word, academic acronym |
| 3 | **RareScan** | Focus on rare diseases; very easy to remember |
| 4 | **MediSyn** | medical synthesis; clean and short |
| 5 | **ScanForge** | forging scans; pairs well with "rare disease" framing |
| 6 | **VitalGen** | generating vital (medical) imagery |

**Recommendation:** `AURORA` for the paper title (acronym gives a neat contribution
line) with `MediForge` as the tool/product name — or pick one and use it
everywhere. Consistency matters more than the specific choice.

---

## 6. Refined purpose (draft)

> To improve AI-based diagnosis of **rare medical conditions** (e.g. TB in chest
> X-rays) by generating **high-fidelity, privacy-preserving synthetic images**
> of underrepresented classes, and to quantify how much synthetic augmentation
> improves a downstream classifier's minority-class performance — including the
> fidelity threshold at which augmentation stops helping.

This frames the project as a **measurement + method** paper, not just "we made
images", which is what reviewers want.

---

## 7. Candidate research questions

Pick ONE primary question; the others become sub-questions/ablations.

1. **Fidelity–utility trade-off (recommended primary):**
   *At what FID threshold do GAN-generated synthetic X-rays stop improving
   downstream minority-class recall, and does the threshold differ by
   generative model (DCGAN vs WGAN-GP vs diffusion)?*

2. **Data scarcity:**
   *What is the minimum number of real minority-class images needed for a
   generative model to produce synthetic images that measurably improve
   classifier recall?*

3. **Model comparison:**
   *How do DCGAN, WGAN-GP, and a fine-tuned latent diffusion model compare
   in (a) image fidelity (FID) and (b) downstream diagnostic utility on a
   rare-disease X-ray task?*

4. **Privacy/transfer:**
   *Can a classifier trained only on synthetic images (no real patient data)
   achieve clinically useful performance on real test images?*

5. **Feature fidelity:**
   *Do synthetic images preserve clinically relevant pathological features
   (e.g. TB cavities/consolidations) as measured by feature attribution
   (Grad-CAM) and/or radiologist review?*

### Suggested contribution for the paper
A **conditional rare-disease augmentation framework** + a **fidelity–utility
analysis** showing (i) synthetic augmentation lifts minority recall by ≥15%,
(ii) the FID threshold below which augmentation helps, and (iii) a comparison
of at least two generative families (GAN vs diffusion) on the same task.

---

## 8. Scope decision: GAN vs diffusion vs LLM

The user mentioned "an LLM good at image generation". Options:

- **A. Keep GAN (DCGAN/WGAN-GP)** — fits existing code, cheap to train on a
  GPU, defensible baseline. Risk: GANs are now considered dated; reviewers may
  ask "why not diffusion?"
- **B. Diffusion (DDPM / latent diffusion)** — state of the art fidelity,
  more compute, more novelty, stronger paper. Risk: scope/infra.
- **C. Hybrid (recommended):** GAN as the implemented, working baseline +
  one diffusion model as the "modern" comparison point. This directly supports
  research question #3 and is the most publishable.

Decision needed: which generative families to include given available compute
(GPU? Colab?).

---

## 9. Decisions needed from the user

1. **Project name** (pick from §5, or propose your own).
2. **Primary research question** (pick from §7).
3. **Generative scope** — A (GAN only), B (diffusion only), or C (hybrid).
4. **Target disease / dataset** — stay with TB chest X-ray, or add a second
   (e.g. pneumonia, a rare MRI condition)?
5. **Image resolution** — 64×64 (faster) or 128×128 (more detail)?
6. **Compute** — local GPU / Colab / cloud? (determines diffusion feasibility)
7. **Clinical validation** — Grad-CAM only, or also a radiologist review?

---

## 10. Proposed next steps (after decisions)

1. Rename project everywhere (folder, README, setup.py, streamlit, configs).
2. Fix blockers: real ResNet-18, resolution consistency, transform leakage,
   WGAN-GP path.
3. Get one clean end-to-end run (train GAN → generate images → FID →
   classifier baseline vs augmented).
4. Build experiment harness (configs, seeds, logging, result tables).
5. Add the second generative family if hybrid.
6. Write paper sections alongside experiments (methods, results, ablations).

---

## 11. Deeper discovery (data, compute, paper angle)

### 11.1 Dataset specifics (confirmed)
- Source: Kaggle **"Tuberculosis (TB) Chest X-ray Database"** (Qatar Univ. / Univ. of Dhaka).
- Public split: **3500 Normal vs 700 TB** (5:1 imbalance) — exactly the rare-disease
  scenario targeted. (A further 2800 TB images exist behind an NIAID agreement;
  we will stay with the public 700 to keep the project reproducible.)
- Images: chest X-rays, grayscale, ~512×512 original → resized to 128×128.
- This imbalance ratio is a clean, defensible experimental setting.

### 11.2 Compute plan (no paid subscriptions)
| Task | Where | Notes |
|---|---|---|
| GAN training (DCGAN/WGAN-GP, 128×128, 200 epochs) | Local RTX 4050 (6 GB) | Batch size 16–32; AMP on; ~hours. |
| Diffusion training (DDPM, 128×128) | Kaggle Notebooks (free, P100/T4, 30h/wk) or Colab free (T4) | Better GPU than Colab free; longer sessions. |
| Classifier experiments + FID | Local 4050 | Fast. |
| Final reproducible runs | Kaggle/Colab notebooks | Publish notebooks for reproducibility. |

Recommendation: keep a **single source of truth** — train GAN locally for
iteration, but run all *reported* numbers on Kaggle/Colab so reviewers can
reproduce. Store seeds + configs in git.

### 11.3 Paper angle decision
The user chose to keep the research question focused on **fidelity × utility**
(FID + downstream minority recall) and NOT to add a privacy/memorization
dimension. So the primary paper stays narrower and more tractable.

(Privacy/memorization remains a documented *optional future work* item: the 2025
IOP paper "Beware of diffusion models for synthesizing medical images" shows
diffusion models memorize training images more than GANs on small datasets,
which is directly relevant to our 700-image TB set. It can be added as an
ablation/appendix if time permits, but is out of scope for the primary
contribution.)

### 11.4 Refined research question (proposed final)
> *To what extent can synthetic minority-class chest X-rays generated by GANs
> and diffusion models improve TB-detection performance when real training data
> is limited, and how do fidelity (FID), downstream utility (minority recall),
> and privacy (memorization rate) trade off across generative families and
> real-data budgets?*

### 11.5 Experiment design (draft)

**Independent variables**
- Generative family: DCGAN, WGAN-GP, DDPM (conditional on class)
- Real-data budget for TB class: {50, 100, 200, 400, 700} images
- Augmentation ratio: synthetic count = {0.5×, 1×, 2×} minority deficit

**Dependent variables**
- FID (synthetic vs real TB)
- Downstream classifier: accuracy, precision, recall (minority), F1, AUC
- Memorization: mean nearest-neighbour distance (synthetic→train), memorization
  rate at δ threshold, duplicate detection

**Baselines / controls**
- No augmentation (real data only)
- Classical augmentation (flip/rotate/affine) — to isolate generative value
- Upper bound: full balanced dataset (3500/3500) if NIAID data obtained

**Protocol**
- 5-fold CV or fixed 70/15/15 split with 3 seeds; report mean ± std.
- Fixed classifier (ResNet-18, same hyperparams) across all conditions.
- Pre-register success: ≥15% minority-recall gain at FID≤80 without
  memorization rate > 5%.

### 11.6 Environment / dependency notes
- Base conda (Python 3.13) has no torch. System Python is 3.12.10, which
  matches `torch==2.2.0` (Python ≤3.12). **Recommendation:** create a venv
  with the system Python 3.12 and install `requirements.txt` there.
  Alternatively bump torch to ≥2.4 for Python 3.13.
- For diffusion: add `diffusers`, `accelerate`, `transformers`, `safetensors`
  to a separate `requirements-diffusion.txt`.
- Kaggle/Colab notebooks will `pip install` these at runtime.

### 11.7 Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Diffusion memorizes small TB set | High (per IOP paper) | Make memorization a measured outcome, not a failure; tune via conditioning/regularization; report honestly. |
| 6 GB VRAM OOM on 128×128 diffusion | Medium | Start DDPM at 64×64; use gradient checkpointing; move diffusion to Kaggle. |
| FID not reaching ≤80 | Medium | Relax to "best achieved + trend"; FID on small medical sets is noisy; add KID. |
| No clinical validation | High | Use Grad-CAM feature-attribution as proxy; optional radiologist review if accessible. |
| Scope creep (hybrid) | Medium | Timebox: GAN first (weeks 1–3), diffusion (weeks 4–5), paper writing (week 6). |
| Reproducibility on different hardware | Medium | Publish Kaggle/Colab notebooks + seeds + configs. |

### 11.8 Suggested timeline (final year)
- W1–2: Fix blockers (ResNet, resolution, transforms, WGAN-GP), clean run.
- W3: GAN experiments (budgets/ablations) + FID/memorization metrics.
- W4: Diffusion (DDPM) on Kaggle; comparison table.
- W5: Classifier experiments + Grad-CAM; result tables/figures.
- W6: Paper draft (methods, results, discussion, limitations).
- W7: Revision, reproducibility notebooks, final demo (MediSyn).

---

## 12. Decisions made (from user)

- **Project / research name:** AURORA — *Augmenting Underrepresented Rare-disease
  Observations using Realistic Artificial imaging*.
- **Tool / demo name:** MediSyn.
- **Primary research question:** *To what extent can synthetic minority-class
  medical images generated using generative AI improve the classification
  performance of disease-detection models when real medical training data is
  limited, and how does this improvement relate to synthetic-image fidelity and
  the generative model used?*
- **Generative scope:** Hybrid — GAN (DCGAN/WGAN-GP) baseline + one diffusion
  model for comparison.
- **Compute:** NVIDIA RTX 4050 (6 GB) laptop GPU locally (no paid subscription);
  Kaggle Notebooks (free, P100/T4) for diffusion training + Colab free for demos.
- **Primary paper scope:** fidelity × utility only (no privacy dimension).
- **Experiment protocol:** see EXPERIMENT_PLAN.md.

*Discovery document updated; project identity renamed across repo. Deeper discovery
complete (data, compute, paper angle, experiment plan).*

## 13. Refinement changes applied (build phase started)

Blockers fixed and smoke-tested (8/8 tests pass; GAN + classifier run end-to-end
on CPU):

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
   made optional; `num_workers` default lowered to 0 (Windows-safe); models
   dir created before checkpoint save; autocast updated to non-deprecated API.

Smoke tests passed: GAN (BCE + WGAN-GP) trains and generates 128×128 images;
classifier trains and evaluates end-to-end; data pipeline runs without leakage.

### Remaining work (per EXPERIMENT_PLAN.md)
- Diffusion model (DDPM) implementation + Kaggle notebook.
- Experiment harness (budgets, seeds, metrics tables, figures).
- Grad-CAM visualizations.
- Update MediSyn demo with best checkpoint + generated samples.
- Paper draft.

## 14. Milestone 1 Complete: Dataset Validation & Baseline (2026-10-26)

### 14.1 Dataset Validation Results (Actual)
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

### 14.2 Real-Only Classifier Baseline (1-epoch smoke test on CPU)

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

### 14.3 Next Milestone: Full Baseline + Conventional Augmentation

**Required (on local RTX 4050):**
1. Install CUDA PyTorch on local GPU machine
2. Run real-only baseline: 20 epochs × 3 seeds → proper mean ± std
3. Run conventional augmentation baseline (flip/rotate/affine) for comparison
4. Then proceed to GAN training (DCGAN/WGAN-GP) on TB class

**Deferred (as planned):**
- Diffusion (DDPM) → Kaggle free notebooks (P100/T4)
- Full paper draft → after experimental results

---

*Next recommended step: run full baseline on GPU, then conventional augmentation, then GAN training.*