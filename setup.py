from setuptools import setup, find_packages

setup(
    name="trading-agent-rl",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12.0",
        "yfinance>=0.2.38",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "scikit-learn>=1.2.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.9",
)
