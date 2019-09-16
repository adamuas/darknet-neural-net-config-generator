import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="darknet_config_generator",
    version="0.0.1",
    author="Abdullahi S. Adamu",
    author_email="abdullah.adam89@gmail.com",
    description="Darknet Neural Network Configuration Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://adamuas.github.io/darknet-neural-net-config-generator/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)