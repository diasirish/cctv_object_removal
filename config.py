import cv2 as cv
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Configuration")
    parser.add_argument("--font", type=int, default=cv.FONT_HERSHEY_SIMPLEX, help="Font type for text")
    parser.add_argument("--font_scale", type=float, default=1.0, help="Font scale for text")
    parser.add_argument("--color", type=int, nargs=3, default=[255, 255, 255], help="Text color (BGR format)")
    parser.add_argument("--thickness", type=int, default=2, help="Thickness of the text")
    parser.add_argument("--label_height", type=int, default=40, help="Height of the label area")
    parser.add_argument("--min_movement_area", type=int, default=500, help="Minimum area of movement to consider")
    
    args = parser.parse_args()
    return args

def get_cfg():
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def update_config_with_args(args):
    global FONT, FONT_SCALE, COLOR, THICKNESS, LABEL_HEIGHT, MIN_MOVEMENT_AREA
    FONT = args.font
    FONT_SCALE = args.font_scale
    COLOR = tuple(args.color)
    THICKNESS = args.thickness
    LABEL_HEIGHT = args.label_height
    MIN_MOVEMENT_AREA = args.min_movement_area

# Default values
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
COLOR = (255, 255, 255)
THICKNESS = 2
LABEL_HEIGHT = 40
MIN_MOVEMENT_AREA = 500

# Update with parsed arguments
args = parse_args()
update_config_with_args(args)