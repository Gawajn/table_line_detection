import glob
import math

import PIL
import numpy as np
from pylsd import lsd
from segmentation.preprocessing.source_image import SourceImage
from table_line_detection.network_postprocessor import TableResult

from table_line_detection.table_lsd_processor import TableLSDProcessorConfig

from PIL import Image, ImageDraw

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


def near_collinear(linea, lineb, angle_thres=1.5, distance_thres=1.25):
    pointa = ((linea[0] + lineb[0]) / 2, (linea[1] + lineb[1]) / 2)
    pointb = ((linea[2] + lineb[2]) / 2, (linea[3] + lineb[3]) / 2)
    mid_angle = line_to_angle([*pointa, *pointb], use_abs=False)
    angle_a, angle_b = line_to_angle(linea, use_abs=False), line_to_angle(lineb, use_abs=False)
    angle_diff_a, angle_diff_b = abs(angle_a - mid_angle), abs(angle_b - mid_angle)
    angle_diff_a = min(angle_diff_a, 180 - angle_diff_a)
    angle_diff_b = min(angle_diff_b, 180 - angle_diff_b)
    print(f"Angles: {angle_diff_a}, {angle_diff_b}")
    if angle_diff_a > angle_thres or angle_diff_b > angle_thres:
        print("c1")

        return False
    mid_point = ((pointa[0] + pointb[0]) / 2, (pointa[1] + pointb[1]) / 2)
    distance_a = distance_line_point(mid_point, linea)
    distance_b = distance_line_point(mid_point, lineb)
    print(f"Distance: {distance_a}, {distance_b} thres: {distance_thres}")
    if distance_a > distance_thres and distance_b > distance_thres:
        print("c2")
        return False
    print("Collinear")
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
    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, width: {image_width}, height: {image_height}, length: {length}")
    print(f"x1_norm: {x1_norm}, y1_norm: {y1_norm}, x2_norm: {x2_norm}, y2_norm: {y2_norm}")
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
                #x1_norm = x1 / img_width
                #x2_norm = x2 / img_width
                #y1_norm = y1 / img_height
                #y2_norm = y2 / img_height
                length = normalized_distance((x1,y1), (x2,y2), image_width=img_width, image_height=img_height) #np.linalg.norm([x1_norm - x2_norm, y1_norm - y2_norm])


                if length > self.config.min_length_to_not_filter_small_line_segments:
                    print("filtered", length, self.config.min_length_to_not_filter_small_line_segments)
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
    #path = "/home/alexanderh/Documents/datasets/gehrke/original/*.png"
    #path = "/home/alexanderh/Documents/datasets/gehrke/original/*.png"
    path = "/tmp/test.png"
    for i in glob.glob(path):

        pil_image = Image.open(i)
        image = np.array(pil_image)
        # image = binarize(image, algorithm=BinarizationAlgorithm("isauvola")).astype("uint8") * 255

        source_image = SourceImage.from_numpy(image)
        processor = TableLSDProcessor()
        output: TableResult = processor.predict_image(source_image)
        full_name = i
