import setuptools

REQUIRED_PACKAGES = [
    'torch',
    'torchvision',
    'lmdb'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glow",
    version="0.1.1",
    author="Emmanuel Fuentes",
    description="PyTorch Generative Flow Modeling Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eifuentes/glow-pytorch",
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=setuptools.find_packages()
)
