#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
from utils import *
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *
import logging

# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--num_distill_iter',
                        type=int,
                        default=500,
                        help="# of iterations for generating a distilled data")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO, filename=f'./res/{args.model}_{args.mp_bit_budget}.log')
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    ###### FP32 model ##########################################
    # Load pretrained model
    logging.info('****** Loading the pretrained FP32 model  ******')
    fp32_model = ptcv_get_model(args.model, pretrained=True)
    logging.info('****** Loaded the pretrained FP32 model  ******')

    # Load validation data
    logging.info('****** Loading the validation data ******')
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              for_inception=args.model.startswith('inception'))
    logging.info('****** Loaded the validation data ******')

    # Generate distilled data
    logging.info('****** Generating a Distilled Data ******')

    dataloader = getDistilData(
        fp32_model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'),
        num_iter=args.num_distill_iter)
    logging.info('****** Generated a Distilled Data ******')

    logging.info('****** Converting to an INT8 model ******')
    # Quantize single-precision model to 8-bit model
    int8_model = quantize_model(fp32_model)
    # Freeze BatchNorm statistics
    int8_model.eval()
    int8_model = int8_model.cuda()
    logging.info('****** Converted to an INT8 model ******')

    logging.info('****** Performing Zero Shot Quantization ******')
    # Update activation range according to distilled data
    update(int8_model, dataloader)
    logging.info('****** Finished Zero Shot Quantization ******')

    # Freeze activation range during test
    freeze_model(int8_model)
    best_config = search_mixed_precision(
        int8_model, dataloader, MODEL_SIZE_MB[args.model] / 32 * args.mp_bit_budget, plot_path=f"./res/{args.model}_{args.mp_bit_budget}.png"
        # a=7, b=100, t0=7, t=3
    )
    logging.info(f'Best Bit Setting: {best_config}')
    unfreeze_model(int8_model)
    update(int8_model, dataloader)
    freeze_model(int8_model)

    int8_model = nn.DataParallel(int8_model).cuda()
    # Test the final quantized model
    logging.info('****** Testing ******')
    test(int8_model, test_loader)
    logging.info('****** Finished Testing ******')
