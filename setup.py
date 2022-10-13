import json
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iaflow",
    version='2.1.15',
    author="Enmanuel Magallanes Pinargote",
    author_email="enmanuelmag@cardor.dev",
    description="This library help to create models with identifiers, checkpoints, logs and metadata automatically, in order to make the training process more efficient and traceable.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enmanuelmag/iaflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'requests',
        'tensorflow',
        'discord_webhook',
        'notifier-function'
    ],
    python_requires='>=3.6',
)
