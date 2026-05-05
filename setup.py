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
    extras_require = {
        "gui": ["PySide6>=6.4"],
        "cli": ["pyyaml>=6.0"],
    },
    entry_points = {
        "console_scripts": [
            # GUI launcher (falls back to CLI if PySide6 is missing)
            "xray_uplim     = xray_uplim.__main__:main",
            # CLI-only launcher (YAML/JSON config file)
            "xray_uplim-cli = xray_uplim.cli:main",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
