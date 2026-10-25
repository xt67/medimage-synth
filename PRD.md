# AURORA - Augmenting Underrepresented Rare-disease Observations using Realistic Artificial imaging

## Overview
AURORA is a generative-AI system that produces high-fidelity synthetic X-ray / MRI
images of underrepresented (rare) disease classes to augment imbalanced medical
imaging datasets. The companion demo app is called MediSyn.

## Core Features
1. **GAN-based Image Generation**: Train DCGAN/WGAN-GP models to generate synthetic medical images
2. **Dataset Augmentation**: Balance imbalanced datasets with synthetic minority class samples
3. **Classifier Improvement**: Demonstrate improved minority class recall after augmentation
4. **Streamlit Demo**: Interactive UI for generating and evaluating synthetic images
5. **Evaluation Metrics**: FID score calculation and classifier performance comparison

## Technical Architecture
- **Data Pipeline**: Load and preprocess medical images (X-rays/MRI)
- **GAN Pipeline**: Generator and discriminator networks with training loops
- **Classifier Pipeline**: ResNet-18 fine-tuning for medical image classification
- **Evaluation**: FID score computation and metrics calculation
- **Demo UI**: Streamlit application for image generation and results visualization

## Implementation Plan (Week-by-Week)
- **Week 1**: Data pipeline setup and testing
- **Week 2**: GAN implementation and initial training
- **Week 3**: GAN optimization and synthetic image generation
- **Week 4**: Classifier implementation and augmentation experiments
- **Week 5**: Streamlit demo creation and final integration

## Success Criteria
- FID score ≤ 80 on synthetic vs real images
- ≥15% improvement in minority-class recall after augmentation
- Working Streamlit demo with image generation capability
- All unit tests passing