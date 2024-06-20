import copy
import glob
import math
import random
from dataclasses import dataclass
from typing import List

import PIL
import numpy as np
import os
from PIL import Image, ImageDraw
from pylsd import lsd
from segmentation.preprocessing.source_image import SourceImage
from table_line_detection.network_postprocessor import TableResult

import math
import numpy as np
@dataclass
class AABB:
    x1: float
    y1: float
    x2: float
    y2: float

    def h(self):
        return self.y2 - self.y1

    def w(self):
        return self.x2 - self.x1

    def intersects(self, other: 'AABB') -> bool:
        return not (other.x1 > self.x2 or other.x2 < self.x1 or other.y1 > self.y2 or other.y2 < self.y1)

    def expand(self, value):
        x, y = value
        self.x1 -= x
        self.y1 -= y
        self.x2 += x
        self.y2 += y

    def copy(self) -> 'AABB':
        return copy.copy(self)
@dataclass
class Point:
    x: float
    y: float


class Line:
    def __init__(self, line: List[Point]):
        self.line: List[Point] = line
        self._aabb: AABB = None

    def aabb(self) -> AABB:
        if not self._aabb:
            x1 = min([p.x for p in self.line])
            y1 = min([p.y for p in self.line])
            x2 = max([p.x for p in self.line])
            y2 = max([p.y for p in self.line])
            self._aabb = AABB(x1, y1, x2, y2)
        return self._aabb

    def get_start_point(self):
        return self.line[0]

    def get_end_point(self):
        return self.line[-1]

    def get_average_line_height(self):
        return np.mean([point.y for point in self.line])

    def get_xy(self):
        x_list = []
        y_list = []
        for point in self.line:
            x_list.append(point.x)
            y_list.append(point.y)
        return x_list, y_list

    def __len__(self):
        return len(self.line)

    def __iter__(self):
        return iter(self.line)

    def __getitem__(self, key):
        return self.line[key]

    def __setitem__(self, key, value):
        self.line[key] = value

    def __str__(self):
        return "[{0}]".format(', '.join(map(str, self.line)))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.line + other.line

    def __radd__(self, other):
        return other.line + self.line

    def l_append(self, value):
        self.line = value + self.line

    def r_append(self, value):
        self.line = self.line + value

    def scale_line(self, factor: float):
        self.line = [Point(point.x * factor, point.y * factor) for point in self.line]

    def __copy__(self):
        return Line(copy.copy(self.line))

    def __delitem__(self, key):
        del self.line[key]

def connect_connected_components_to_line(cc_list: List[Line], staff_line_height: int,
                                         staff_space_height: int) -> List[Line]:
    def connect_cc(cc_list: List[Line]) -> List[Line]:
        def prune_cc(cc_list: List[Line], length: int) -> List[Line]:
            pruned_cc_list = []
            for cc in cc_list:
                if abs(cc.get_start_point().x - cc.get_end_point().x) > length:
                    pruned_cc_list.append(cc)
            return pruned_cc_list

        def connect(max_dists: List[int], vert_dist: int, cc_list: List[Line]) -> List[Line]:
            for max_dist in max_dists:
                i = 0
                while i < len(cc_list):
                    l1 = cc_list[i].line
                    p1_b = l1[0]
                    p1_e = l1[-1]

                    found = False
                    for i2 in range(i + 1, len(cc_list)):
                        l2 = cc_list[i2].line
                        p2_b = l2[0]
                        p2_e = l2[-1]
                        if p1_e.x < p2_b.x and p2_b.x - p1_e.x < max_dist:
                            if np.abs(p1_e.y - p2_b.y) < vert_dist:
                                cc_list[i].line = l1 + l2
                                del cc_list[i2]
                                found = True
                                break
                        elif p2_e.x < p1_b.x and p1_b.x - p2_e.x < max_dist:
                            if np.abs(p1_b.y - p2_e.y) < vert_dist:
                                cc_list[i].line = l2 + l1
                                del cc_list[i2]
                                found = True
                                break
                    if not found:
                        i += 1
                if vert_dist == 2 and max_dist == 30:
                    cc_list = prune_cc(cc_list, 10)
            return cc_list

        cc_list_copy = cc_list

        for x in [[10, 30, 50, 100], [200, 300, 500]]:
            for vert_dist in [2, staff_line_height, staff_space_height / 5 + staff_line_height,
                              staff_space_height / 3 + staff_line_height]:
                cc_list_copy = connect(x, vert_dist, cc_list_copy)
        return cc_list_copy

    llc = connect_cc(cc_list)
    return llc

@dataclass
class TableLSDProcessorConfig:
    max_parallel_distance_thresh: float = 8
    max_angle_thresh = 1.5
    max_merge_distance_thresh = 0.05
    min_length_to_not_filter_small_line_segments = 0.05

def merge_line_segments(linesegments, config: TableLSDProcessorConfig):
    for i in range(len(linesegments)):
        near_collinear = []
        for j in range(i + 1, len(linesegments)):
            if linesegments[i].intersects(linesegments[j]):
                linesegments[i] = linesegments[i].union(linesegments[j])
                del linesegments[j]
    pass

def line_to_angle(line, use_abs=True):
    x1, y1, x2, y2 = line[:4]
    if abs(x2 - x1) <= 0.0000001:
        angle_val = 90 if y2 > y1 else -90
    else:
        angle_val = np.degrees(math.atan((y2 - y1) / (x2 - x1)))
    if use_abs:
        angle_val = abs(angle_val)
    return angle_val


def distance_line_point(p, line):
    p1 = np.array(p)
    p2, p3 = np.array(line[:2]), np.array(line[2:4])
    return np.abs(np.cross(p2-p1, p1-p3) / np.linalg.norm(p2-p1))


def near_collinear(linea, lineb, angle_thres=1.5, distance_thres=1.5):
    pointa = ((linea[0] + lineb[0]) / 2, (linea[1] + lineb[1]) / 2)
    pointb = ((linea[2] + lineb[2]) / 2, (linea[3] + lineb[3]) / 2)
    mid_angle = line_to_angle([*pointa, *pointb], use_abs=False)
    angle_a, angle_b = line_to_angle(linea, use_abs=False), line_to_angle(lineb, use_abs=False)
    angle_diff_a, angle_diff_b = abs(angle_a - mid_angle), abs(angle_b - mid_angle)
    angle_diff_a = min(angle_diff_a, 180 - angle_diff_a)
    angle_diff_b = min(angle_diff_b, 180 - angle_diff_b)
    #print(f"Angles: {angle_diff_a}, {angle_diff_b}")
    if angle_diff_a > angle_thres and angle_diff_b > angle_thres:
        #print("c1")

        return False
    mid_point = ((pointa[0] + pointb[0]) / 2, (pointa[1] + pointb[1]) / 2)
    distance_a = distance_line_point(mid_point, linea)
    distance_b = distance_line_point(mid_point, lineb)
    #print(f"Distance: {distance_a}, {distance_b} thres: {distance_thres}")
    if distance_a > distance_thres and distance_b > distance_thres:
        #print("c2")
        return False
    #print("Collinear")
    return True

def sqrt_distance_of_two_line_segments(line1, line2):
    pass

def normalized_distance(p1, p2, image_width, image_height):
    x1, y1 = p1
    x2, y2 = p2
    x1_norm = x1 / image_width
    x2_norm = x2 / image_width
    y1_norm = y1 / image_height
    y2_norm = y2 / image_height
    length = np.linalg.norm([x1_norm - x2_norm, y1_norm - y2_norm])
    #print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, width: {image_width}, height: {image_height}, length: {length}")
    #print(f"x1_norm: {x1_norm}, y1_norm: {y1_norm}, x2_norm: {x2_norm}, y2_norm: {y2_norm}")
    return length #np.linalg.norm([p1[0] / image_width - p2[0] / image_width, p1[1] / image_height - p2[1] / image_height])

class TableLSDProcessor:
    def __init__(self, config: TableLSDProcessorConfig = TableLSDProcessorConfig()):
        self.config = config

        pass

    def predict_image(self, img: SourceImage, keep_dim: bool = True, processes: int = 1) -> PIL.Image:

        img_gray = img.get_grayscale_array()

        segments = lsd(img_gray, scale=0.5)

        img_color = img.pil_image.convert('RGB')
        draw = ImageDraw.Draw(img_color)
        img_height, img_width = img_gray.shape
        l_segments = []
        def filter_line_segments(segments):
            filtered_segments = []
            for i in range(segments.shape[0]):
                x1, y1, x2, y2, width = segments[i]
                length = normalized_distance((x1,y1), (x2,y2), image_width=img_width, image_height=img_height) #np.linalg.norm([x1_norm - x2_norm, y1_norm - y2_norm])

                if length > self.config.min_length_to_not_filter_small_line_segments:
                    #print("filtered", length, self.config.min_length_to_not_filter_small_line_segments)
                    filtered_segments.append(segments[i])
            return np.array(filtered_segments)
        segments = filter_line_segments(segments)
        line_list = []
        line_list_tuples = []
        for i in range(segments.shape[0]):
            line_list_tuples.append((int(segments[i, 0]), int(segments[i, 1]), int(segments[i, 2]), int(segments[i, 3])))
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))
            width = segments[i, 4]
            color = tuple(np.random.randint(256, size=3))
            draw.line((pt1, pt2), fill=color, width=int(np.ceil(width / 2)))

        start = 0
        restart = True

        while restart:
            restart = False
            for l1 in range(start, len(line_list_tuples)):
                if restart:
                    break
                start = l1

                restart = False
                x1, y1, x2, y2 = line_list_tuples[l1]
                #print(f"Line: {x1}, {y1}, {x2}, {y2}")
                #print("l1", l1, len(line_list_tuples))
                for l2 in range(l1 + 1, len(line_list_tuples)):
                    #merged = LineMerger().merge_lines2(line_list[l1], line_list[l2])
                    x3, y3, x4, y4 = line_list_tuples[l2]
                    img_color2 = img.pil_image.convert('RGB')
                    draw2 = ImageDraw.Draw(img_color2)
                    color = tuple(np.random.randint(256, size=3))
                    draw2.line(((x1, y1), (x2, y2)), fill=(255, 0, 0), width=int(np.ceil(3)))
                    draw2.line(((x3, y3), (x4, y4)), fill=(0, 255, 0), width=int(np.ceil(3)))
                    #distance = sqrt_distance_of_two_line_segments(line_list_tuples[l1], line_list_tuples[l2])
                    def min__distance_between_two_line_segs(line1, line2):
                        x1, y1, x2, y2 = line1
                        x3, y3, x4, y4 = line2
                        p1 = (x1, y1)
                        p2 = (x2, y2)
                        p3 = (x3, y3)
                        p4 = (x4, y4)
                        noramlized_min_distance = min(normalized_distance(p1, p3, image_width=img_width, image_height=img_height),
                                                      normalized_distance(p1, p4, image_width=img_width, image_height=img_height),
                                                      normalized_distance(p2, p3, image_width=img_width, image_height=img_height),
                                                      normalized_distance(p2, p4, image_width=img_width, image_height=img_height))


                        if x1 < x3 and x2 > x4:
                            return noramlized_min_distance
                        if x1 < x3 and x4 > x2:
                            return noramlized_min_distance
                        if x3 < x1 and x4 < x2:
                            return noramlized_min_distance
                        return noramlized_min_distance

                    if min__distance_between_two_line_segs(line_list_tuples[l1], line_list_tuples[l2]) > self.config.max_merge_distance_thresh:
                        #print("distance")
                        continue
                    collinear = near_collinear(line_list_tuples[l1], line_list_tuples[l2], angle_thres=self.config.max_angle_thresh, distance_thres=self.config.max_parallel_distance_thresh)

                    if collinear:
                        #print("merged")
                        restart = True

                        x_val = [x1, x2, x3, x4]
                        y_val = [y1, y2, y3, y4]
                        min_x = min(x_val)
                        max_x = max(x_val)
                        min_y = min(y_val)
                        max_y = max(y_val)
                        dif_x = max_x - min_x
                        dif_y = max_y - min_y
                        # pt1 = i.point1()
                        # pt2 = i.point2()



                        if dif_x > dif_y:
                            line_list_tuples[l1] = (min_x, y_val[x_val.index(min_x)], max_x, y_val[x_val.index(max_x)])
                            draw2.line(((min_x, y_val[x_val.index(min_x)]+5), (max_x, y_val[x_val.index(max_x)]+5)), fill=(0, 0, 255), width=int(np.ceil(3)))

                        else:
                            line_list_tuples[l1] = (x_val[y_val.index(min_y)], min_y, x_val[y_val.index(max_y)],max_y)
                            draw2.line(((x_val[y_val.index(min_y)] + 5, min_y), (x_val[y_val.index(max_y)] + 5,max_y)), fill=(0, 0, 255), width=int(np.ceil(3)))

                        del line_list_tuples[l2]
                        from matplotlib import pyplot as plt
                        #plt.imshow(np.array(img_color2))
                        #plt.show()
                        break
                    from matplotlib import pyplot as plt
                    #plt.imshow(np.array(img_color2))
                    #plt.show()
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
        img_color2 = img.pil_image.convert('RGB')
        draw2 = ImageDraw.Draw(img_color2)
        for i in line_list_tuples:
            x1, y1, x2, y2 = i
            #pt1 = i.point1()
            #pt2 = i.point2()
            color = tuple(np.random.randint(256, size=3))
            draw2.line(((x1,y1), (x2,y2)), fill=color, width=int(np.ceil(6)))
        ax[0].imshow(np.array(img_color))
        ax[1].imshow(np.array(img_color2))
        plt.show()
        pass
        if keep_dim:
            return #TableResult()
        else:
            return #TableResult()


if __name__ == "__main__":



    #exit()
    path = "/home/alexanderh/Documents/datasets/gehrke/original/*.png"
    for i in glob.glob(path):

        pil_image = Image.open(i)
        image = np.array(pil_image)
        # image = binarize(image, algorithm=BinarizationAlgorithm("isauvola")).astype("uint8") * 255

        source_image = SourceImage.from_numpy(image)
        processor = TableLSDProcessor()
        output: TableResult = processor.predict_image(source_image)
        full_name = i
