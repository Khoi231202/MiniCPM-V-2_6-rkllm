#!/usr/bin/env python
# coding: utf-8

import os
from rknn.api import RKNN
from sys import exit
import argparse
import cv2
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

image_sizes= [[448, 448]]
batch_sizes = [1]

def convert_encoder():
    rknn = RKNN(verbose=True)

    ONNX_MODEL=f"vision_transformer.onnx"
    RKNN_MODEL=ONNX_MODEL.replace(".onnx",".rknn")
    DATASET="dataset.txt"
    QUANTIZE=False
    input_shapes = [[[batch_size, 3, image_size[0], image_size[1]]] for batch_size in batch_sizes for image_size in image_sizes]
    print(input_shapes)

    # pre-process config
    print('--> Config model')
    rknn.config(quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588', optimization_level=3,
                mean_values=[128, 128, 128], std_values=[128, 128, 128], dynamic_input=input_shapes) # mean_values=[0.5, 0.5, 0.5], std_values=[0.5, 0.5, 0.5],
    print('done')

    # Load ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
    )

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE, dataset=DATASET, rknn_batch_size=None)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # export
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')
    rknn.init_runtime(target='rk3588')
    # # image embedding
    # img_path = "test.jpg"

    # normalize_mean = [0.5, 0.5, 0.5]
    # normalize_std = [0.5, 0.5, 0.5]

    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (448, 448))
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
    # # img = (img - normalize_mean) / normalize_std
    # img = img[np.newaxis, :, :, :]
    # img = img.transpose(0, 3, 1, 2)
    # np.save("img.npy", img)
    # rknn.accuracy_analysis(inputs=["img.npy"], target='rk3588')
# usage: python convert_rknn.py encoder|all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model to convert", choices=["encoder", "all"], nargs='?')
    args = parser.parse_args()
    if args.model is None:
        args.model = "all"
    if args.model == "encoder":
        convert_encoder()
    elif args.model == "all":
        convert_encoder()
    else:
        print(f"Unknown model: {args.model}")
        exit(1)
