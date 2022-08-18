#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

import trainer
from exp import get_exp
from utils import configure_nccl, configure_omp, get_num_devices

_SUPPORTED_DATASETS = ["coco", "linemod", "coco_kpts"]
_NUM_CLASSES = {"coco":80, "linemod":15, "coco_kpts":1}
_VAL_ANN = {
    "coco":"instances_val2017.json", 
    "linemod":"instances_test.json",
    "coco_kpts": "person_keypoints_val2017.json",
}
_TRAIN_ANN = {
    "coco":"instances_train2017.json", 
    "linemod":"instances_train.json",
    "coco_kpts": "person_keypoints_train2017.json",
}
_SUPPORTED_TASKS = {
    "coco":["2dod"],
    "linemod":["2dod", "6dpose"],
    "coco_kpts": ["2dod", "human_pose"]
}

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=0, type=int, help="device for training"
    )
    parser.add_argument(
        "-w", "--workers", default=None, type=int, help="number of workers per gpu"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("--dataset", default="coco_kpts", type=str, help="dataset for training")
    parser.add_argument("--task", default="coco_kpts", type=str, help="type of task for model eval")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        default=False,
        action="store_true",
        help="Enable drawing of bounding cuboid"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    if args.dataset is not None:
        assert (
            args.dataset in _SUPPORTED_DATASETS
        ), "The given dataset is not supported for training!"
        exp.data_set = args.dataset
        exp.num_classes = _NUM_CLASSES[args.dataset]
        exp.val_ann = _VAL_ANN[args.dataset]
        exp.train_ann = _TRAIN_ANN[args.dataset]

        if args.task is not None:
            assert (
                args.task in _SUPPORTED_TASKS[args.dataset]
            ), "The specified task cannot be performed with the given dataset!"
            if args.dataset == "linemod":
                #exp.pose = True if args.task == "6dpose" else exp.pose = False
                if args.task == "6dpose":
                    exp.object_pose = True
            elif args.dataset=="coco_kpts":
                if args.task == "human_pose":
                    exp.human_pose=True

        if args.visualize:
            exp.visualize = args.visualize

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    main()