"""
PyTorch vision transforms / data augmentation. 
"""


class VisionAugment:
    def __init__(
        self,
    ):  # TODO: Vision augmentation pipeline, define functions outside of class.
        raise NotImplementedError


class VisionDataset:
    def __init__(
        self, file_type: str
    ):  # TODO: Lazy loading for locally stored image datasets.
        self.file_type = file_type
        raise NotImplementedError
