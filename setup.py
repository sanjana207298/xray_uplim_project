from setuptools import setup, find_packages

setup(
    name             = "xray_uplim",
    version          = "2.0.0",
    author           = "Sanjana Gupta",
    description      = "Unified X-ray non-detection upper limit calculator (NuSTAR, XMM-Newton, Swift, Chandra)",
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
            "xray-uplim=xray_uplim.nustar.pipeline:run_uplim",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
