from setuptools import setup, find_packages

setup(
    name             = "nustar_uplim",
    version          = "1.0.0",
    author           = "Sanjana Gupta",
    description      = "NuSTAR count-rate upper limits for non-detections",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "numpy>=1.21",
        "scipy>=1.7",
        "astropy>=5.0",
        "matplotlib>=3.4",
    ],
    entry_points = {
        "console_scripts": [
            "nustar-uplim=nustar_uplim.pipeline:run_uplim",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
