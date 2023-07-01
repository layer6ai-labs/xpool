import os
from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSVDDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file = 'data/MSVD/captions_msvd.json'
        db_file_negative = 'data/MSVD/noun_random.json'
        db_file_negative_verb = 'data/MSVD/verb_random.json'
        test_file = 'data/MSVD/test_list_small.txt'
        train_file = 'data/MSVD/train_list_small.txt'
        self.vid2caption = load_json(db_file)
        self.vid2Negativecaption_noun = load_json(db_file_negative)
        self.vid2Negativecaption_verb = load_json(db_file_negative_verb)

        if split_type == 'train':
            self.train_vids = read_lines(train_file) 
            self._construct_all_train_pairs()
        else:
            self.test_vids = read_lines(test_file)
            self._construct_all_test_pairs()


    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id, negativeNoun, negCaptionVerb = self._get_vidpath_and_caption_by_index_train(index)
        else:
            video_path, caption, video_id, negativeNoun, negCaptionVerb = self._get_vidpath_and_caption_by_index_test(index)

        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)
        
        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
            'neg_noun': negativeNoun,
            'neg_verb': negCaptionVerb
        }

        return ret


    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption, negativeNoun, negCaptionVerb = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid, negativeNoun, negCaptionVerb

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption, negativeNoun, negCaptionVerb = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid, negativeNoun, negCaptionVerb

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)


    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            for caption_idx in range(len(self.vid2caption[vid])):
                # self.all_test_pairs.append([vid, vid2caption[vid][caption_idx], self.vid2Negativecaption[vid][caption_idx]])
                self.all_train_pairs.append([vid, self.vid2caption[vid][caption_idx], self.vid2Negativecaption_noun[vid][caption_idx], self.vid2Negativecaption_verb[vid][caption_idx]])


    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            for caption_idx in range(len(self.vid2caption[vid])):
                self.all_test_pairs.append([vid, self.vid2caption[vid][caption_idx], self.vid2Negativecaption_noun[vid][caption_idx], self.vid2Negativecaption_verb[vid][caption_idx]])
