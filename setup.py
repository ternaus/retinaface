import os
import re
import sys
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple

from setuptools import Command, find_packages, setup

# Package meta-data.
name = "retinaface_pytorch"
description = "Retinaface implementation in Pytorch."
url = "https://github.com/ternaus/retinaface"
email = "iglovikov@gmail.com"
author = "Vladimir Iglovikov"
requires_python = ">=3.0.0"
current_dir = Path(__file__).absolute().parent


def get_version() -> str:
    version_file = current_dir / "retinaface" / "__init__.py"
    with version_file.open(encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)  # type: ignore


# What packages are required for this module to be executed?
try:
    with (current_dir / "requirements.txt").open(encoding="utf-8") as f:
        required = f.read().split("\n")
except FileNotFoundError:
    required = []

# What packages are optional?
extras = {"test": ["pytest"]}

version = get_version()

about = {"__version__": version}


def get_test_requirements() -> List[str]:
    requirements = ["pytest"]
    if sys.version_info < (3, 3):
        requirements.append("mock")
    return requirements


def get_long_description() -> str:
    base_dir = Path(__file__).absolute().parent
    with (base_dir / "README.md").open(encoding="utf-8") as f:
        return f.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options: List[Tuple] = []

    @staticmethod
    def status(s: str) -> None:
        """Print things in bold."""
        print(s)  # noqa: T001

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        try:
            self.status("Removing previous builds...")
            rmtree(os.path.join(current_dir, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        os.system(f"{sys.executable} setup.py sdist bdist_wheel --universal")

        self.status("Uploading the package to PyPI via Twine...")
        os.system("twine upload dist/*")

        self.status("Pushing git tags...")
        os.system(f"git tag v{about['__version__']}")
        os.system("git push --tags")

        sys.exit()


setup(
    name=name,
    version=version,
    description=description,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Vladimir Iglovikov",
    license="MIT",
    url=url,
    packages=find_packages(exclude=["tests", "docs", "images"]),
    install_requires=required,
    extras_require=extras,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={"upload": UploadCommand},
)
