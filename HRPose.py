import os

from exp import Exp as MyExp
import torch
import torch.distributed as dist
import torch.nn as nn

import models
from config import cfg
from config import update_config
import torch
import torch.backends.cudnn as cudnn

gpu = 0
# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

def update_config_from_file(cfg, args_file_path):
    cfg.defrost()
    cfg.merge_from_file(args_file_path)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
                cfg.DATA_DIR, cfg.DATASET.ROOT
                )

    cfg.freeze()


class HRNet_Anchor(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        #transition_out, merge_out, hrnet_outs = self.backbone(x)
        hrnet_outs = self.backbone(x)
        if self.training:
            assert targets is not None
            if isinstance(self.head, models.yolo_kpts_head_64.YOLOXHeadKPTS):
                loss, iou_loss, conf_loss, cls_loss, l1_loss, kpts_loss, kpts_vis_loss, num_fg = self.head(
                        hrnet_outs, targets, x
                        )
                outputs_H = {
                        "total_loss": loss,
                        "iou_loss": iou_loss,
                        "l1_loss": l1_loss,
                        "conf_loss": conf_loss,
                        "cls_loss": cls_loss,
                        "kpts_loss": kpts_loss,
                        "kpts_vis_loss": kpts_vis_loss,
                        "num_fg": num_fg,
                        }    
        else:
            outputs_H = self.head(hrnet_outs)
        return outputs_H

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname( __file__ )), "coco_kpts")

        # -----------------  testing config ------------------ #
        self.human_pose = True

    def get_model(self):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels=3
        '''
        extra=dict(
            stem=dict(  
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(4, 4, 4),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (48, 96),
                    (48, 96, 192),
                    (48, 96, 192, 384),
                )),
            with_head=False,
            )
        '''
        extra=dict(
                stem=dict(  
                          stem_channels=32,
                          out_channels=32,
                          expand_ratio=1),
                num_stages=3,
                stages_spec=dict(
                    num_modules=(3, 8, 3),
                    num_branches=(2, 3, 4),
                    num_blocks=(4, 4, 4),
                    module_type=('LITE', 'LITE', 'LITE'),
                    with_fuse=(True, True, True),
                    reduce_ratios=(8, 8, 8),
                    num_channels=(
                        (48, 96),
                        (48, 96, 192),
                        (48, 96, 192, 384),
                        )),
                    with_head=False,
                    )

        #hrnet_model = models.hrnet.get_pose_net(cfg, is_train=True)
        backbone = models.ditehrhrnet.DiteHRNet(extra)
        #torch.cuda.set_device(gpu)
        #backbone = hrnet_model.cuda(gpu)
        #print(hrnet_model)
        in_channels = [48, 96, 192, 384]
        #in_channels = [128, 256, 512]
        head = models.yolo_kpts_head_64.YOLOXHeadKPTS(self.num_classes, self.width, in_channels=in_channels)

        self.model = HRNet_Anchor(backbone, head)
        #self.model = self.model.to('cuda')
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
            ):
        from data import (
                COCOKPTSDataset,
                TrainTransform,
                YoloBatchSampler,
                DataLoader,
                InfiniteSampler,
                MosaicDetection,
                worker_init_reset_seed,
                )

        if self.data_set == "coco_kpts":
            dataset = COCOKPTSDataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob),
                    cache=cache_img,
                    )


        dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    object_pose=self.object_pose,
                    human_pose=self.human_pose,
                    flip_index=dataset.flip_index,
                    ),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,

                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
                )
        #            perspective=self.perspective,
        self.dataset = dataset

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=False,
                mosaic=not no_aug,
                )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from data import COCODataset, COCOKPTSDataset, LINEMODDataset, ValTransform

        if self.data_set == "coco_kpts":
            valdataset = COCOKPTSDataset(
                    data_dir=self.data_dir,
                    json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                    name="val2017" if not testdev else "test2017",
                    img_size=self.test_size,
                    preproc=ValTransform(legacy=legacy),
                    human_pose = self.human_pose
                    )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                    valdataset, shuffle=False
                    )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
                "num_workers": self.data_num_workers,
                "pin_memory": True,
                "sampler": sampler,
                }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        if self.human_pose:
            evaluator = COCOEvaluator(
                    dataloader=val_loader,
                    img_size=self.test_size,
                    confthre=self.test_conf,
                    nmsthre=self.nmsthre,
                    num_classes=self.num_classes,
                    testdev=testdev,
                    )
        return evaluator
