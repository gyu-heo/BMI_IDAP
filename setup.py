from pathlib import Path

from distutils.core import setup
import copy

dir_parent = Path(__file__).parent

def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements

deps_all = read_requirements()

## Get version number
with open(str(dir_parent / "bmi_idap" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break


setup(
    name='bmi_idap',
    packages=['bmi_idap'],
    version=version,
    description="BMI interday analysis pipeline",
    author='Rich Hakim',
    license='MIT',
    install_requires=deps_all,
)

# from distutils.core import setup

# setup(
#     name='BMI_IDAP',
#     version='0.1.0',
#     packages=['bmi_idap',],
#     license='MIT',
#     long_description=open('README.md').read(),
# )