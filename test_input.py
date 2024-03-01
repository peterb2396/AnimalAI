import os
import torch
import string
import random
import numpy as np
from utils.utils import read_config
from models.timesformerclipinitvideoguide import (
    TimeSformerCLIPInitVideoGuide,
)
from utility import extract_frames as ef
import torchvision


print("here")
inp = ef.video_to_frames('parrot.avi','Output_Video')

model_args = (
    #train_loader,
    #val_loader,
    #criterion,
    #eval_metric,
    #class_list,
    #args.test_every,
    #args.distributed,
    #device,
)
# Amphibian = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
# Amphibian.load("./checkpoint_Amphibian.pth")
# Bird = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
# Bird.load("./checkpoint_Bird.pth")
Reptile = TimeSformerCLIPInitVideoGuide()
Reptile.load("./checkpoint_Reptile.pth")
# Fish = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
# Fish.load("./checkpoint_Fish.pth")
# Mammal = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
# Mammal.load("./checkpoint_Mammal.pth")
# Insect = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
# Insect.load("./checkpoint_Insect.pth")
# Sea_Animal = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
# Sea_Animal.load("./checkpoint_Sea-animal.pth")
test1 = Reptile(inp)
print(test1)
