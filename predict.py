#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: release2.7 
# @Product: PyCharm 
# @Time: 2023/8/31 17:04:54
# @Author: 图灵的猫
# @File: predict
import os
import cv2

import ppocr.utility as utility
from ppocr.predict_system import TextSystem

ocr = TextSystem(utility.parse_args())

image = os.path.join('imgs/lite_demo.png')
img = cv2.imread(image)

det_boxes, rec_res, _ = ocr(img)
# print(det_boxes, rec_res)
[print(text) for text in rec_res]
