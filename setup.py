from setuptools import setup, find_packages

setup(
    name="kubeflow-101",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "kfp>=1.8.0",
        "kubeflow-pipeline-sdk>=1.8.0",
        "xgboost>=1.7.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "mlflow>=2.0.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0.0",
        "kubernetes>=26.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Kubeflow pipeline for XGBoost classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kubeflow-101",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
) 