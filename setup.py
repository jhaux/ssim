from setuptools import setup, find_packages


# allows to get version via python setup.py --version
__version__ = "0.1.0"


setup(
    name="ssim_callback",
    version=__version__,
    description="SSIM callback for edflow eval pipeline",
    url="https://github.com/jhaux/ssim",
    author="Johannes Haux et al",
    author_email="johannes.haux@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "scikit-image>=0.15",
        "edflow"
    ],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
