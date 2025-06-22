from setuptools import find_packages, setup

from STGas import __author__, __author_email__, __homepage__, __version__

if __name__ == "__main__":
    setup(
        name="STGas",
        version=__version__,
        url=__homepage__,
        author=__author__,
        author_email=__author_email__,
        keywords="deep learning",
        packages=find_packages(exclude=("config", "tools", "demo")),
        classifiers=[
            "Development Status :: Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        license="Apache License 2.0",
        zip_safe=False,
    )
