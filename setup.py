from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyqtb",
    version="0.1",
    url="https://github.com/PQCLab/pyQTB",
    author="Boris Bantysh",
    author_email="bbantysh60000@gmail.com",
    description="Python library for benchmarking quantum tomography methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={"pyqtb": ["utils/mubs.pickle"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.2",
        "scipy>=1.6.2",
        "dill>=0.3.4",
        "py-cpuinfo>=8.0.0"
    ],
)
