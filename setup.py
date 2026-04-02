from setuptools import setup, find_packages

setup(
    name="medimage-synth",
    version="1.0.0",
    description="Generative AI for Synthetic Medical Image Augmentation",
    author="Rayan",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "Pillow>=10.2.0",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.1",
        "matplotlib>=3.8.3",
        "seaborn>=0.13.2",
        "plotly>=5.19.0",
        "streamlit>=1.32.0",
        "pytorch-fid>=0.3.0",
        "PyYAML>=6.0.1",
        "tensorboard>=2.16.2",
        "tqdm>=4.66.2",
        "pytest>=8.1.1",
        "gdown>=5.1.0",
    ],
    python_requires=">=3.10",
)