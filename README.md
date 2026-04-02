# MedImage Synth

**Generative AI for Synthetic Medical Image Augmentation**

Final Year B.Tech Project | Data Science / AI-ML

## Overview

MedImage Synth is a GAN-based system designed to generate high-fidelity synthetic X-ray or MRI images for rare diseases. Its primary purpose is to augment imbalanced medical imaging datasets so that downstream classification or detection models can be trained with less bias and higher accuracy - without ever requiring access to real patient data.

## Features

- Privacy-preserving: No real patient data is shared or stored
- GAN-based synthetic image generation (DCGAN/WGAN-GP)
- Quantitative evaluation with FID scores
- Downstream classifier improvement demonstration
- Streamlit demo UI

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training the GAN

```bash
python src/gan/train_gan.py --config configs/dcgan.yaml
```

### Running the Demo

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
medimage-synth/
├── data/                    # Datasets
│   ├── raw/                # Original images
│   ├── processed/          # Preprocessed images
│   └── synthetic/          # GAN-generated images
├── models/                 # Saved model checkpoints
├── src/
│   ├── data_loader.py      # Data loading & preprocessing
│   ├── gan/                # GAN modules
│   ├── classifier/         # Classification modules
│   ├── evaluate/           # Evaluation metrics
│   └── utils/              # Utilities
├── configs/                # YAML configs
├── app/                    # Streamlit demo
└── tests/                  # Unit tests
```

## Dataset

This project uses publicly available datasets:
- Kaggle TB Dataset (recommended for MVP)
- NIH Chest X-ray14
- CheXpert (Stanford)

## License

For research and educational purposes only.