import multiprocessing

import numpy as np
import scipy

from segmentation.util import angle_to, PerformanceCounter, logger

import math


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


class BaseLineCCs(object):
    def __init__(self, cc, type):
        self.cc = cc
        index_min = np.where(cc[1] == min(cc[1]))  # [0]
        index_max = np.where(cc[1] == max(cc[1]))  # [0]

        if type == 'baseline':
            self.cc_left = (np.mean(cc[0][index_min][0]), cc[1][index_min[0]][0])
            self.cc_right = (np.mean(cc[0][index_max][0]), cc[1][index_max[0]][0])

        else:
            self.cc_left = (np.mean(cc[0]), cc[1][index_min[0]][0])
            self.cc_right = (np.mean(cc[0]), cc[1][index_max[0]][0])

        self.type = type

    def __lt__(self, other):
        return self.cc < other


def calculate_distance(index, ccs, maximum_angle, baseline_border_image):
    print(baseline_border_image)
    index1 = []
    index2 = []
    value = []
    x = ccs[index]
    ind1 = index
    for ind2 in range(index, len(ccs)):  # 0
        y = ccs[ind2]
        if x is y:
            index1.append(ind1)
            index2.append(ind2)
            value.append(0)
        else:
            distance = 0
            same_type = 1 if x.type == y.type else 1000

            def left(x, y):
                return x.cc_left[1] > y.cc_right[1]

            def right(x, y):
                return x.cc_right[1] < y.cc_left[1]

            if left(x, y):
                angle = angle_to(np.array(y.cc_right), np.array(x.cc_left))
                distance = x.cc_left[1] - y.cc_right[1]
                #height_difference = abs(x.cc_left[0] - y.cc_right[0])
                print(f"distance {distance}")

                test_angle = maximum_angle if distance > 30 else maximum_angle * 5 if distance > 5 else maximum_angle * 10
                print(f"ltest angle {test_angle} angle: {angle}")

                if test_angle < angle < (360 - test_angle):
                    if distance < 0 and abs(distance) < 5 and abs(x.cc_left[0] - y.cc_right[0]) < 5:
                        distance = abs(distance)
                    else:
                        distance = 99999
                    ## small overlap
                    distance = 99999

                else:

                    if baseline_border_image:
                        point_c = y.cc_right
                        point_n = x.cc_left

                        x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                        y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                             [point_c[0], point_n[0]]).astype(int)
                        indexes = (y_points, x_points)
                        blackness = np.sum(baseline_border_image[indexes])
                        # print('left' + str(blackness))
                        distance = distance * (blackness * 5000 + 1)

            elif right(x, y):
                angle = angle_to(np.array(x.cc_right), np.array(y.cc_left))
                distance = y.cc_left[1] - x.cc_right[1]
                print(f"distance {distance}")
                test_angle = maximum_angle if distance > 30 else maximum_angle * 5 if distance > 5 else maximum_angle * 10
                #height_difference = abs(y.cc_left[0] - x.cc_right[0])
                print(f"rtest angle {test_angle} angle: {angle}")
                if test_angle < angle < (360 - test_angle):
                    if distance < 0 and abs(distance) < 5 and abs(x.cc_left[0] - y.cc_right[0]) < 5:
                        distance = abs(distance)
                    else:
                        distance = 99999
                    distance = 99999
                else:

                    if baseline_border_image:
                        point_c = x.cc_right
                        point_n = y.cc_left

                        x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                        y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                             [point_c[0], point_n[0]]).astype(
                            int)
                        indexes = (y_points, x_points)

                        blackness = np.sum(baseline_border_image[indexes])
                        distance = distance * (blackness * 5000 + 1)
            else:
                distance = 99999
            index1.append(ind1)
            index2.append(ind2)
            value.append(distance * same_type)
            index1.append(ind2)
            index2.append(ind1)
            value.append(distance * same_type)
            # distance_matrix[ind1, ind2] = distance * same_type
    return (index1, index2), value


def calculate_distance2(index, ccs, max_delta_y, baseline_border_image):
    index1 = []
    index2 = []
    value = []
    x = ccs[index]
    ind1 = index
    for ind2 in range(index, len(ccs)):  # 0
        y = ccs[ind2]
        if x is y:
            index1.append(ind1)
            index2.append(ind2)
            value.append(0)
        else:
            distance = 0
            same_type = 1 if x.type == y.type else 1000

            def left(x, y):
                return x.cc_left[1] < y.cc_left[1]

            def right(x, y):
                return x.cc_right[1] > y.cc_left[1]

            if left(x, y):
                #angle = angle_to(np.array(y.cc_right), np.array(x.cc_left))
                distance = abs(x.cc_right[1] - y.cc_left[1])
                #height_difference = abs(x.cc_left[0] - y.cc_right[0])
                y_delta = abs(x.cc_left[0] - y.cc_right[0])

                if y_delta > max_delta_y:
                    distance = 99999

                if baseline_border_image:
                    point_c = y.cc_right
                    point_n = x.cc_left

                    x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                    y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                         [point_c[0], point_n[0]]).astype(int)
                    indexes = (y_points, x_points)
                    blackness = np.sum(baseline_border_image[indexes])
                    # print('left' + str(blackness))
                    distance = distance * (blackness * 5000 + 1)

            else:
                distance = abs(y.cc_right[1] - x.cc_left[1])

                y_delta = abs(y.cc_left[0] - x.cc_right[0])

                if y_delta > max_delta_y:
                    distance = 99999

                if baseline_border_image:
                    point_c = x.cc_right
                    point_n = y.cc_left

                    x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                    y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                         [point_c[0], point_n[0]]).astype(
                        int)
                    indexes = (y_points, x_points)

                    blackness = np.sum(baseline_border_image[indexes])
                    distance = distance * (blackness * 5000 + 1)

            index1.append(ind1)
            index2.append(ind2)
            value.append(distance * same_type)
            index1.append(ind2)
            index2.append(ind1)
            value.append(distance * same_type)
            # distance_matrix[ind1, ind2] = distance * same_type
    return (index1, index2), value

def extracted_ccs_optimized(array: np.array):
    raveled_array = array.ravel()

    index = np.arange(len(raveled_array))
    sort_idx = np.argsort(raveled_array)
    cnt = np.bincount(raveled_array)
    res = np.split(index[sort_idx], np.cumsum(cnt[:-1]))[1:]
    ccs = [np.unravel_index(res[ind], array.shape) for ind, i in enumerate(res)]
    return ccs


def extract_tables_from_probability_map(image_map: np.array, line_horizontal_index=1, line_vertical_index=2,
                                        original=None, processes=8, predict_borders=False):
    image = np.argmax(image_map, axis=-1)

    return extract_table_lines(image_map=image, line_horizontal_index=line_horizontal_index,
                             line_vertical_index=line_vertical_index, original=original, processes=processes,
                             connection_width=100, predict_borders=predict_borders)


def extract_horizontal_lines(image_map: np.array, line_horizontal_index=1, line_vertical_index=2, original=None,
                             processes=1, connection_width=100):
    lines = extract_table_lines(image_map, line_horizontal_index, line_vertical_index, original, processes,
                              connection_width)
    pass


def extract_table_lines(image_map: np.array, line_horizontal_index=1, line_vertical_index=2, original=None, processes=1,
                      connection_width=100, predict_borders=False, db_scan_delta_distance=5):
    from scipy.ndimage.measurements import label

    base_ind = np.where(image_map == line_horizontal_index)
    base_border_ind = np.where(image_map == line_vertical_index)

    baseline = np.zeros(image_map.shape)
    baseline[base_ind] = 1
    if not predict_borders:
        baseline_border = None
    else:
        baseline_border = np.zeros(image_map.shape)

        baseline_border[base_border_ind] = 1
    baseline_ccs, n_baseline_ccs = label(baseline, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    baseline_ccs = extracted_ccs_optimized(baseline_ccs)

    #baseline_ccs = [np.where(baseline_ccs == x) for x in range(1, n_baseline_ccs + 1)]
    baseline_ccs = [BaseLineCCs(x, 'baseline') for x in baseline_ccs if len(x[0]) > 50]

    all_ccs = baseline_ccs  # + baseline_border_ccs
    logger.info("Extracted {} CCs from probability map \n".format(len(all_ccs)))
    """"""

    def calculate_distance_matrix(ccs, maximum_angle=5, processes=8):
        distance_matrix = np.zeros((len(ccs), len(ccs)))

        from functools import partial
        distance_func = partial(calculate_distance2, ccs=ccs, max_delta_y=maximum_angle, baseline_border_image=baseline_border
                                )
        indexes_ccs = list(range(len(ccs)))
        if processes is not None and processes > 1:
            with multiprocessing.Pool(processes=processes, maxtasksperchild=100) as p:
                out = list(p.map(distance_func, indexes_ccs))
        else:
            out = list(map(distance_func, indexes_ccs))
        for x in out:
            indexes, values = x
            distance_matrix[indexes] = values
        return distance_matrix

    with PerformanceCounter(function_name="calculate_distance_matrix"):
        matrix = calculate_distance_matrix(all_ccs, maximum_angle=db_scan_delta_distance, processes=processes)

    from sklearn.cluster import DBSCAN
    if np.sum(matrix) == 0:
        print("Empty Image")
        return
    t = DBSCAN(eps=connection_width, min_samples=1, metric="precomputed").fit(matrix)

    ccs = []
    for x in np.unique(t.labels_):
        ind = np.where(t.labels_ == x)
        line = []
        for d in ind[0]:
            if all_ccs[d].type == 'baseline':
                line.append(all_ccs[d])
        if len(line) > 0:
            ccs.append((np.concatenate([x.cc[0] for x in line]), np.concatenate([x.cc[1] for x in line])))

    ccs = [list(zip(x[0], x[1])) for x in ccs]

    from itertools import chain
    from typing import List, Tuple
    from collections import defaultdict

    def normalize_connected_components(cc_list: List[List[Tuple[int, int]]]):
        # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
        def normalize(point_list):
            normalized_cc_list = []
            for cc in point_list:
                cc_dict = defaultdict(list)
                for y, x in cc:
                    cc_dict[x].append(y)
                normalized_cc = []
                for key in sorted(cc_dict.keys()):
                    value = cc_dict[key]
                    normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
                normalized_cc_list.append(normalized_cc)
            return normalized_cc_list

        return normalize(cc_list)

    ccs = normalize_connected_components(ccs)
    new_ccs = []
    for baseline in ccs:
        new_ccs.append([coord_tup[::-1] for coord_tup in baseline])

    return new_ccs
