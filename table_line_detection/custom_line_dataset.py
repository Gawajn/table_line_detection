import math
import os
from enum import IntEnum
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from dataclasses import dataclass

from albumentations.pytorch import ToTensorV2
from mashumaro.mixins.json import DataClassJSONMixin
from segmentation.util import show_images
from torch.utils.data import Dataset, DataLoader

from segmentation.callback import ModelWriterCallback
from segmentation.datasets.dataset import get_rescale_factor, rescale_pil, dirs_to_pandaframe
from segmentation.losses import Losses
from segmentation.metrics import Metrics
from segmentation.model_builder import ModelBuilderMeta, ModelBuilderLoad
from segmentation.modules import Architecture
from segmentation.network import NetworkTrainer
from segmentation.optimizer import Optimizers
from segmentation.preprocessing.workflow import PreprocessingTransforms, NetworkEncoderTransform, GrayToRGBTransform, \
    ColorMapTransform, BinarizeDoxapy
import json
import albumentations
import cv2
from albumentations import RandomScale, RandomGamma, RandomBrightnessContrast, OneOf, ToGray, CLAHE, Compose, Affine, \
    ShiftScaleRotate, ImageCompression, JpegCompression
import albumentations as albu
from segmentation.scripts.train import get_default_device
from segmentation.settings import ColorMap, ClassSpec, Preprocessingfunction, PredefinedNetworkSettings, \
    CustomModelSettings, ModelConfiguration, ProcessingSettings, NetworkTrainSettings


def get_line_border(p1, p2, length=20, max_width=None, max_height=None, lw=5):
    x1, y1 = p1
    x2, y2 = p2

    x3 = x2 - x1
    y3 = y2 - y1
    lw = lw + 1
    if x2 > x1:
        if max_width > x2 + lw / 2:
            x2 = x2 + lw / 2
    if x2 < x1:
        if 0 < x2 - lw / 2:
            x2 = x2 - lw / 2
    mag = math.sqrt(x3 * x3 + y3 * y3)
    x3 = x3 / mag
    y3 = y3 / mag

    temp = x3
    x3 = -y3
    y3 = temp

    xl1 = x2 + x3 * length
    xl2 = x2 + x3 * -length

    yl1 = y2 + y3 * length
    yl2 = y2 + y3 * -length
    return Line([Point(xl1, yl1), Point(xl2, yl2)])


def default_transform():
    result = Compose([
        ShiftScaleRotate(rotate_limit=1, scale_limit=(-0.1, 0.1), shift_limit_x=0.2, shift_limit_y=0.2,
                         border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), mask_value=0),
        #Affine(shear=2, cval=(255, 255, 255), cval_mask=0),
        albumentations.HorizontalFlip(p=0.25),
        RandomGamma(),
        ImageCompression(quality_lower=5, quality_upper=100, p=0.5),
        RandomBrightnessContrast(),
        albumentations.OneOf([
            albumentations.OneOf([
                BinarizeDoxapy("sauvola"),
                BinarizeDoxapy("ocropus"),
                BinarizeDoxapy("isauvola"),
            ]),
            albumentations.OneOf([
                albumentations.ToGray(),
                albumentations.CLAHE()
            ])
        ], p=0.0)

    ])
    return result


def remove_nones(x):
    return [y for y in x if y is not None]


class TableLabel(IntEnum):
    BACKGROUND = 0
    HORIZONTAL_LINES = 1
    VERTICAL_LINES = 2
    HORIZONTAL_LINES_BORDER = 3
    VERTICAL_LINES_BORDER = 4

    def get_color(self):
        return {0: [255, 255, 255],
                1: [255, 0, 0],
                2: [0, 255, 0],
                3: [255, 0, 255],
                4: [0, 255, 255]

                }[self.value]


@dataclass
class LineDrawConfig:
    line_width: int = 5
    horizontal_lines_color: Tuple[int] = (255, 0, 0)
    vertical_lines_color: Tuple[int] = (0, 255, 0)
    draw_borders: bool = False
    line_border_width: int = 5
    line_border_length: int = 5
    horizontal_lines_color_border: Tuple[int] = (255, 0, 255)
    vertical_lines_color_border: Tuple[int] = (0, 255, 255)
    draw_first: str = "vertical"


@dataclass
class Point(DataClassJSONMixin):
    x: int
    y: int


@dataclass
class Line(DataClassJSONMixin):
    line: List[Point]

    def to_list(self, rescale_factor: float = 1.0):
        return [(int(x.x * rescale_factor), int(x.y) * rescale_factor) for x in self.line]

    def draw_line(self, image: Image, width, color, rescale_factor: float = 1.0):
        ImageDraw.Draw(image).line(self.to_list(rescale_factor=rescale_factor),
                                   fill=color,
                                   width=width)


@dataclass
class SimpTableLineFileFormat(DataClassJSONMixin):
    horizontal_lines: List[Line]
    vertical_lines: List[Line]
    pass

    def draw_borders(self, image: Image, config: LineDrawConfig, rescale_factor: float = 1.0):
        for i in self.horizontal_lines:
            get_line_border((i.line[0].x, i.line[0].y), (i.line[-1].x, i.line[-1].y), config.line_border_length,
                            image.width, image.height).draw_line(image, config.line_border_width,
                                                                 config.horizontal_lines_color_border, rescale_factor)
            get_line_border((i.line[-1].x, i.line[-1].y), (i.line[0].x, i.line[0].y),
                            config.line_border_length, image.width, image.height).draw_line(image,
                                                                                            config.line_border_width,
                                                                                            config.horizontal_lines_color_border,
                                                                                            rescale_factor)
        for i in self.vertical_lines:
            get_line_border((i.line[0].x, i.line[0].y), (i.line[-1].x, i.line[-1].y), config.line_border_length,
                            image.width, image.height).draw_line(image, config.line_border_width,
                                                                 config.vertical_lines_color_border, rescale_factor)
            get_line_border((i.line[-1].x, i.line[-1].y), (i.line[0].x, i.line[0].y), config.line_border_length,
                            image.width, image.height).draw_line(image, config.line_border_width,
                                                                 config.vertical_lines_color_border, rescale_factor)

    def draw_horizontal_lines(self, image: Image, config: LineDrawConfig, rescale_factor: float = 1.0):
        for i in self.horizontal_lines:
            i.draw_line(image, config.line_width, config.horizontal_lines_color, rescale_factor)

    def draw_vertical_lines(self, image: Image, config: LineDrawConfig, rescale_factor: float = 1.0):
        for i in self.vertical_lines:
            i.draw_line(image, config.line_width, config.vertical_lines_color, rescale_factor)

    def draw_all_lines(self, image: Image, config: LineDrawConfig, rescale_factor: float = 1.0):
        if config.draw_first == "vertical":
            self.draw_vertical_lines(image, config, rescale_factor)
            self.draw_horizontal_lines(image, config, rescale_factor)
            if config.draw_borders:
                self.draw_borders(image, config, rescale_factor)
        else:
            self.draw_horizontal_lines(image, config, rescale_factor)
            self.draw_vertical_lines(image, config, rescale_factor)
            if config.draw_borders:
                self.draw_borders(image, config, rescale_factor)


def crop_mask_image(image, max_pixel_size):
    current_pixel_size = image.width * image.height

    aspect_ratio_w_h = image.width / image.height
    ratio = max_pixel_size / current_pixel_size
    if current_pixel_size > max_pixel_size:
        new_w = math.floor(image.width * ratio)
        new_h = math.floor(image.height * ratio)
        range_w = image.width - new_w
        range_h = image.height - new_h
        start_w = random.randint(0, range_w)
        start_h = random.randint(0, range_h)
        return (start_w, start_w + new_w), (start_h, start_h + new_h)

    return (0, image.width), (0, image.height)


class TableDataset(Dataset):
    def __init__(self, df,
                 transforms: PreprocessingTransforms = None, scale_area=2000000):
        self.df = df
        self.transforms = transforms
        self.index = self.df.index.tolist()
        self.scale_area = scale_area

    def __getitem__(self, item, apply_preprocessing=True):
        image_id, mask_id = self.df.get('images')[item], self.df.get('masks')[item]
        image = Image.open(image_id)
        # rescale_factor = get_rescale_factor(image, scale_area=self.scale_area)
        rescale_factor = 1.0
        # print(f"imageId: {image_id}, maskId: {mask_id}, rescale_factor: {rescale_factor}")
        # image = rescale_pil(image, rescale_factor, 1)
        # print(rescale_factor)
        pil_image = Image.new('RGB', (image.width, image.height), (255, 255, 255))

        SimpTableLineFileFormat.from_dict(json.load(open(mask_id))).draw_all_lines(pil_image, config=LineDrawConfig(),
                                                                                   rescale_factor=rescale_factor)
        show_images([image, pil_image], ["Image", "Mask"])

        x, y = crop_mask_image(image, self.scale_area)

        mask = np.array(pil_image)

        # mask = self.mask_generator.get_mask(mask_id, rescale_factor)
        image = np.array(image)
        image = image[y[0]:y[1], x[0]:x[1], :]
        mask = mask[y[0]:y[1], x[0]:x[1], :]

        if image.dtype == bool:
            image = image.astype("uint8") * 255

        transformed = self.transforms.transform_train(image, mask)  # TODO: switch between modes based on parameter
        if False:
            image_tf = transformed["image"].cpu().numpy().transpose([1, 2, 0])
            std = np.array([0.229, 0.224, 0.225])
            mean = np.array([0.485, 0.456, 0.406])
            image_tf = image_tf * std + mean
            image_tf = image_tf * 255
            image_tf = image_tf.astype("uint8")
            show_images([image, image_tf, transformed["mask"].cpu().numpy() * 255], ["Original", "Augmented", "Mask"])
        return transformed["image"], transformed["mask"], torch.tensor(item)

    def __len__(self):
        return len(self.index)


"""
augmentation = default_transform()
architecture: Architecture = Architecture.UNET
encoder: str = 'efficientnet-b3'
custom_model: bool = False
predefined_encoder_depth = PredefinedNetworkSettings.encoder_depth
predefined_decoder_channel = PredefinedNetworkSettings.decoder_channel  # (256, 256, 196, 128, 64) #(256, 128, 64, 32, 16) #(256, 256, 196, 128, 64)
use_batch_norm_layer = True
custom_model_encoder_filter = [16, 32, 64, 256, 512]
custom_model_decoder_filter = [16, 32, 64, 256, 512]
custom_model_encoder_depth = CustomModelSettings.encoder_depth
"""


def train():
    predfined_nw_settings = PredefinedNetworkSettings(
        architecture=Architecture.UNET,
        encoder='efficientnet-b7',
        classes=len(TableLabel),
        encoder_depth=PredefinedNetworkSettings.encoder_depth,
        decoder_channel=PredefinedNetworkSettings.decoder_channel,
        use_batch_norm_layer=False)
    custom_nw_settings = CustomModelSettings(
        encoder_filter=[16, 32, 64, 256, 512],
        decoder_filter=[16, 32, 64, 256, 512],
        attention_encoder_filter=[12, 32, 64, 128],
        attention=CustomModelSettings.attention,
        classes=len(color_map),
        attention_depth=CustomModelSettings.attention_depth,
        encoder_depth=CustomModelSettings.encoder_depth,
        attention_encoder_depth=CustomModelSettings.attention_encoder_depth,
        stride=CustomModelSettings.stride,
        padding=CustomModelSettings.padding,
        kernel_size=CustomModelSettings.kernel_size,
        weight_sharing=False if CustomModelSettings.weight_sharing else True,
        scaled_image_input=CustomModelSettings.scaled_image_input
    )
    use_custom_model = False
    config = ModelConfiguration(use_custom_model=use_custom_model,
                                network_settings=predfined_nw_settings if not use_custom_model else None,
                                custom_model_settings=custom_nw_settings if use_custom_model else None,
                                preprocessing_settings=ProcessingSettings(input_padding_value=32,
                                                                          rgb=True,
                                                                          scale_max_area=999999999,
                                                                          preprocessing=Preprocessingfunction() if not use_custom_model else Preprocessingfunction(
                                                                              "efficientnet-b7"),
                                                                          transforms=transforms.to_dict()),

                                color_map=color_map)
    network = ModelBuilderMeta(config, device=get_default_device()).get_model()
    return network, config


def finetune():
    base_model_file = ModelBuilderLoad.from_disk(
        model_weights="/home/alexanderh/Documents/datasets/table/models/best.torch", device=get_default_device())
    base_model = base_model_file.get_model()
    base_config = base_model_file.get_model_configuration()
    return base_model, base_config


if __name__ == "__main__":
    doc1 = "/home/alexanderh/Documents/datasets/table/doc/"
    gt1 = "/home/alexanderh/Documents/datasets/table/gt/"
    doc2 = "/home/alexanderh/Documents/datasets/table/doc2/"
    gt2 = "/home/alexanderh/Documents/datasets/table/gt2/"
    doc3 = "/home/alexanderh/Documents/datasets/table/SynthTableDS3/img/"
    gt3 = "/home/alexanderh/Documents/datasets/table/SynthTableDS3/masks/"
    #images_dirs = [doc3]
    #mask_dir = [gt3]
    images_dirs = ["/home/alexanderh/Documents/datasets/table/doc2/"]
    mask_dir = ["/home/alexanderh/Documents/datasets/table/gt2/"]

    df = dirs_to_pandaframe(images_dirs, mask_dir)

    color_map = ColorMap([ClassSpec(label=i.value, name=i.name.lower(), color=i.get_color()) for i in TableLabel])
    input_transforms = Compose(remove_nones([
        GrayToRGBTransform() if True else None,
        ColorMapTransform(color_map=color_map.to_albumentation_color_map())

    ]))
    aug_transforms = default_transform()
    tta_transforms = None
    post_transforms = Compose(remove_nones([
        NetworkEncoderTransform(Preprocessingfunction.name),
        NetworkEncoderTransform(
            'efficientnet-b7'),
        ToTensorV2()
    ]))
    transforms = PreprocessingTransforms(
        input_transform=input_transforms,
        aug_transform=aug_transforms,
        # tta_transforms=tta_transforms,
        post_transforms=post_transforms,
    )
    train_data = TableDataset(df=df, transforms=transforms.get_train_transforms())
    val_data = TableDataset(df=df, transforms=transforms.get_test_transforms())
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    network, config, = train()
    #network, config, = finetune()

    mw = ModelWriterCallback(network, config, save_path=Path("/tmp/"), prefix="",
                             metric_watcher_index=0)
    callbacks = [mw]
    trainer = NetworkTrainer(network, NetworkTrainSettings(classes=len(color_map),
                                                           optimizer=Optimizers("adam"),
                                                           learningrate_seghead=1e-4,
                                                           learningrate_encoder=1e-4,
                                                           learningrate_decoder=1e-4,
                                                           batch_accumulation=1,
                                                           processes=1,
                                                           metrics=[Metrics("accuracy")],
                                                           watcher_metric_index=0,
                                                           loss=Losses.cross_entropy_loss,
                                                           ), get_default_device(),
                             callbacks=callbacks, debug_color_map=config.color_map)

    os.makedirs(os.path.dirname("/tmp/"), exist_ok=True)
    trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, n_epoch=50, lr_schedule=None)
    pass
