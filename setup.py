from setuptools import find_packages, setup

setup(
    name="portal_analysis",
    version="0.2.0",
    description="Booth hand movement classification — training and inference",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "aeon",
        "joblib",
        "tensorflow",
        "mediapipe",
        "opencv-python",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "portal-analysis=portal_analysis.cli:main",
        ],
    },
)
