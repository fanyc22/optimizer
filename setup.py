from setuptools import find_packages, setup


setup(
    name="codesign-optimizer",
    version="0.1.0",
    description="Two-Stage Hardware-Software Co-Design Optimizer for SuperPOD architectures.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.7",
        "typer>=0.12",
        "rich>=13.7",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "codesign-opt=codesign_optimizer.cli:app",
        ],
    },
)
