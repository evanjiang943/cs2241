from setuptools import setup, find_packages

setup(
    name="graphsum",
    version="0.1.0",
    description="Graph Property-Preserving Summarization Framework",
    author="Evan Jiang, Ryan Jiang, Roy Han",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.6.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "python-louvain>=0.15",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)