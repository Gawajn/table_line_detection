import abc
from dataclasses import dataclass
from enum import Enum

import numpy as np

from table_line_detection.postprocessing.table_extraction import extract_table_lines


class TableLinePostProcessor(Enum):
    DbScanPostProcessor = "dbscanPostProcessor"

    def get_class(self) -> "PostProcessor":
        return {
            "dbscanPostProcessor": DbScanPostProcessor
        }[self.value]


@dataclass
class TablePostProcessorConfig:
    algorithm = TableLinePostProcessor.DbScanPostProcessor
    dbscan_merge_distance: int = 10

    processes: int = 8

class PostProcessor(abc.ABC):
    def __init__(self, config: TablePostProcessorConfig):
        self.config = config

    @abc.abstractmethod
    def extract_lines(self, image_map):
        pass


class DbScanPostProcessor(PostProcessor):
    def __init__(self, config: TablePostProcessorConfig):
        super().__init__(config)

    def extract_lines(self, image_map, line_horizontal_index=2, line_vertical_index=4):
        image = np.argmax(image_map, axis=-1)
        from matplotlib import pyplot as plt
        plt.imshow(image)
        plt.show()
        return extract_table_lines(image_map=image, line_horizontal_index=line_horizontal_index,
                                   line_vertical_index=line_vertical_index, original=None,
                                   processes=self.config.processes, connection_width=self.config.dbscan_merge_distance,
                                   predict_borders=None)


