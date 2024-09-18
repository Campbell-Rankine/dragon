from setuptools import setup, find_packages

setup(
    name="dragon",
    version="0.1.0",  # must match the github version
    author="Campbell Rankine",
    author_email="campbellrankine@gmail.com",
    description="A pytorch integrated Machine Learning / Deep learning utilities library",
    packages=find_packages(),
    install_requires=[  # I get to this in a second
        "validators",
        "beautifulsoup4",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License Version 2.0",
        "Operating System :: OS Independent",
        "Pytorch :: Pytorch deep learning utilities",
    ],
    python_requires=">=3.10",
)
