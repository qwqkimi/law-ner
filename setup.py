import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='law-ner',
    author='Terry Luke',
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="law-ner package",
)
