from enum import Enum


class DatasetMode(Enum):
    singlefile = 'singlefile'
    multifile = 'multifile'
    coco = 'coco'

    def __str__(self):
        return self.value
