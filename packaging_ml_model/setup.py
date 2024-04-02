import io
import os
from pathlib import Path

from setuptools import find_packages, setup

# METADATA of our package
NAME:str = "prediction_model"
DESCRIPTION:str = "Loan Prediciton Model"
URL:str = "https://github.com/subratamondal1/loan_eligiblity"
EMAIL:str = "subratasubha2@gmail.com"
AUTHOR:str = "Subrata Mondal"
REQUIRES_PYTHON = ">=3.11.0"

pwd:str = os.path.abspath(path=os.path.dirname(p=__file__))

# Get the list of packages to be installed
def list_requirements(fname="requirements.txt"):
    with io.open(file=os.path.join(pwd,fname), encoding="utf-8") as f:
        return f.read().splitlines()
    
try:
    with io.open(file=os.path.join(pwd,"README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description =  DESCRIPTION

# Load the package's __version__.py module as a dictionary
ROOT_DIR:Path = Path(__file__).resolve().parent
PACKAGE_DIR:Path = ROOT_DIR/NAME
about:dict = {}

with open(file=PACKAGE_DIR/"VERSION") as f:
    _version:str = f.read().strip()
    about["__version__"] = _version

setup(
    name = NAME,
    version = about["__version__"],
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author = AUTHOR,
    author_email = EMAIL,
    python_requires = REQUIRES_PYTHON,
    url = URL,
    packages = find_packages(exclude=("tests",)),
    package_data = {"prediction_model":["VERSION"]},
    install_requires = list_requirements(),
    extras_require = {},
    include_package_data = True,
    license = "MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],

)