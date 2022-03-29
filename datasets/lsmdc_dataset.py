import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class LSMDCDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        
        self.clip2caption = {}
        if split_type == 'train':
            train_file = 'data/LSMDC/LSMDC16_annos_training.csv'
            self._compute_clip2caption(train_file)
               
        else:
            test_file = 'data/LSMDC/LSMDC16_challenge_1000_publictect.csv'
            self._compute_clip2caption(test_file)
  

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    
    def __len__(self):
        return len(self.clip2caption)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        clip_id = list(self.clip2caption.keys())[index]
        caption = self.clip2caption[clip_id]
        clip_prefix = clip_id.split('.')[0][:-3]
        video_path = os.path.join(self.videos_dir, clip_prefix, clip_id + '.avi')

        return video_path, caption, clip_id

            
    def _compute_clip2caption(self, csv_file):
        with open(csv_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, _, _, _, _, caption = line_split
                if clip_id == '1012_Unbreakable_00.05.16.065-00.05.21.941':
                    continue
                self.clip2caption[clip_id] = caption
