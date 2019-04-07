import os
import glob
import random

from torchvision.datasets.folder import pil_loader

import consts

from model import Net
from utils import pil_to_model_tensor_transform

# UTKFace constants

MALE = 0
FEMALE = 1

WHITE = 0
BLACK = 1
ASIAN = 2
INDIAN = 3
OTHER = 4

dset_path = os.path.join('.', 'data', 'UTKFace', 'unlabeled')
rstdir = 'output'

consts.NUM_Z_CHANNELS = 100
net = Net()
load_path = "trained_models/100_Z_channels_200th_epoch"
net.load(load_path, slim=True)  # slim tells the net to load only the encoder and generator

# Set the attributes of a random person you want to test
age = 19
gender = FEMALE
race = ASIAN

image_path = random.choice(glob.glob(os.path.join(dset_path, '{a}_{g}_{r}*'.format(a=age, g=gender, r=race))))
image_tensor = pil_to_model_tensor_transform(pil_loader(image_path))
net.test_single(image_tensor=image_tensor, age=age, gender=gender, target=rstdir, watermark=False)
