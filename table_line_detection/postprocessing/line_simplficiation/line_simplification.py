import abc
from enum import Enum

import numpy as np

from table_line_detection.postprocessing.line_simplficiation.ramer_douglas import ramerdouglas
from table_line_detection.postprocessing.line_simplficiation.visvalingam import VWSimplifier


class LineSimplificationProcessor(Enum):
    NoPostProcessor = "no_post_processor"
    RamerDouglasProcessor = "ramer_douglas"
    VisvalingamProcessor = "visvalingam"


    def get_class(self) -> "LineSimplificationPostProcessor":
        return {
            "ramer_douglas": RamerDouglasProcessor,
            "visvalingam": VisvalingamProcessor

        }[self.value]


class LineSimplificationPostProcessor(abc.ABC):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def simplify_lines(self, line_list):
        pass


class RamerDouglasProcessor(LineSimplificationPostProcessor):

    def __init__(self, config):
        super().__init__(config)

    def simplify_lines(self, line_list):

        simplified_lines = []
        for line_segment in line_list:
            simplified = ramerdouglas(line_segment, dist=self.config.ramer_dogulas_dist)
            simplified_lines.append(simplified)
        return simplified_lines


class VisvalingamProcessor(LineSimplificationPostProcessor):
    def __init__(self, config):
        super().__init__(config)

    def simplify_lines(self, line_list):

        simplified_lines = []
        for line_segment in line_list:
            line_seg = np.asarray(line_segment, dtype=np.float64)
            simplifier = VWSimplifier(line_seg)
            simplified = simplifier.from_number(self.config.max_points_vw)
            simplified_lines.append(simplified)
        return simplified_lines

