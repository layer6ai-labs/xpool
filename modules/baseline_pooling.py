import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

"""
Inputs (assume L2 normalized)
    text_embeds: num_texts x embed_dim
    video_embeds: num_vids x num_frames x embed_dim
"""

class BaselinePooling(nn.Module):
    def __init__(self, pooling_type, config: Config):
        super(BaselinePooling, self).__init__()
        
        assert pooling_type is not None, \
                'Need to specify pooling type when using baseline model.'

        if pooling_type == 'avg':
            self.pooling_func = self._avg_pooling
            print("Using average pooling")
        
        elif pooling_type == 'topk':
            self.k = config.k
            assert self.k > 0
            self.pooling_func = self._topk_pooling
            print("Using top-{} frame pooling".format(self.k))

        elif pooling_type == 'attention':
            self.temperature = config.attention_temperature
            assert self.temperature > 0.0
            self.pooling_func = self._attention_pooling
            print("Using attention pooling")
        
        else:
            raise NotImplementedError

    
    def _avg_pooling(self, text_embeds, video_embeds):
        """
        Pooling mean of frames

        Output
            video_embeds_pooled: num_vids x embed_dim
        """
        video_embeds_pooled = video_embeds.mean(dim=1)
        return video_embeds_pooled

    
    def _topk_pooling(self, text_embeds, video_embeds):
        """
        Pooling top-k frames for each video based on
        similarities with each text query
        
        Output
            video_embeds_pooled: num_vids x num_texts x embed_dim
        """
        num_texts, embed_dim = text_embeds.shape

        # num_vids x num_frames x num_texts
        sims = video_embeds @ text_embeds.t()
        sims_topk = torch.topk(sims, self.k, dim=1)[1]

        # Make format compatible with torch.gather
        video_embeds = video_embeds.unsqueeze(-1).expand(-1, -1, -1, num_texts)
        sims_topk = sims_topk.unsqueeze(2).expand(-1, -1, embed_dim, -1)

        # num_vids x k x embed_dim x num_texts
        video_embeds_topk = torch.gather(video_embeds, dim=1, index=sims_topk)
        
        # Top-k pooling => num_vids x embed_dim x num_texts
        video_embeds_pooled = video_embeds_topk.sum(dim=1)
        return video_embeds_pooled.permute(0,2,1)


    def _attention_pooling(self, text_embeds, video_embeds):
        """
        Pooling frames for each video using attention-based
        similarity with each text query

        Output
            video_embeds_pooled: num_vids x num_texts x embed_dim
        """
        # num_vids x num_frames x num_texts
        sims = video_embeds @ text_embeds.t()
        attention_weights = F.softmax(sims/self.temperature, dim=1)
        
        # num_vids x embed_dim x num_frames
        video_embeds = video_embeds.permute(0,2,1)

        # num_vids x embed_dim x num_texts
        video_embeds_pooled = torch.bmm(video_embeds, attention_weights)
        return video_embeds_pooled.permute(0,2,1)


    def forward(self, text_embeds, video_embeds):
        return self.pooling_func(text_embeds, video_embeds)
