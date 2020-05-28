from setuptools import setup

__version__ = "0.0.1"

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

setup(
    name="pyed",
    version=__version__,
    install_requires=requirements,
    author="Marc Harper",
    author_email="marcharper@gmail.com",
    packages=["evodyn"],
    license="The MIT License (MIT)",
    description="Evolutionary Dynamics Trajectories",
    long_description_content_type="text/x-rst",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.5",
    package_data={'pyed': ['bomze.txt']},
)
