import abc
from dataclasses import dataclass
from enum import Enum

import numpy as np

from table_line_detection.postprocessing.line_simplficiation.line_simplification import LineSimplificationProcessor
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
    line_vertical_index = 2
    line_horizontal_index = 1
    dbscan_merge_distance: int = 25
    dbscan_delta: int = 13
    smooth_lines_algorithm: LineSimplificationProcessor = LineSimplificationProcessor.VisvalingamProcessor
    ramer_dogulas_dist: float = 0.5
    max_points_vw: int = 5
    minimum_line_length = 20
    processes: int = 8

class PostProcessor(abc.ABC):
    def __init__(self, config: TablePostProcessorConfig):
        self.config = config


    @abc.abstractmethod
    def extract_lines_algorithm(self, image_map):
        pass

    def extract_lines(self, image_map, line_index, border_index):
        lines = self.extract_lines_algorithm(image_map, line_index, border_index)
        if  self.config.smooth_lines_algorithm == self.config.smooth_lines_algorithm.NoPostProcessor:
            return lines
        else:
            line_simplification = self.config.smooth_lines_algorithm.get_class()(self.config)
            simplified_lines = line_simplification.simplify_lines(lines)
            simplified_linesv2 = []
            for line in simplified_lines:
                lenght = abs(line[0][0] - line[-1][0])
                if lenght > self.config.minimum_line_length:
                    simplified_linesv2.append(line)

            return simplified_linesv2


class DbScanPostProcessor(PostProcessor):
    def __init__(self, config: TablePostProcessorConfig):
        super().__init__(config)

    def extract_lines_algorithm(self, image_map, line_index, border_index):
        image = np.argmax(image_map, axis=-1)
        return extract_table_lines(image_map=image, line_horizontal_index=line_index,
                                   line_vertical_index=border_index, original=None,
                                   processes=self.config.processes, connection_width=self.config.dbscan_merge_distance,
                                   predict_borders=None, db_scan_delta_distance=self.config.dbscan_delta)



