from setuptools import setup, find_packages

setup(
    name="dragon",
    version="0.0.1",
    author="Campbell Rankine",
    author_email="campbellrankine@gmail.com",
    description="A pytorch integrated Machine Learning / Deep learning utilities library",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License Version 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
