from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 
        self.batch_size = config.batch_size
        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def cosine_similarity(self, x1, x2):
        return torch.mm(x1, x2.transpose(0, 1))

    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            # data['text'] +=  data['neg_text']
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
                data['neg_noun'] = self.tokenizer(data['neg_noun'], return_tensors='pt', padding=True,
                                              truncation=True)
                data['neg_verb'] = self.tokenizer(data['neg_verb'], return_tensors='pt', padding=True,
                                              truncation=True)
                                              
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
                data['neg_noun'] = data['neg_noun'].to(self.device)
                data['neg_verb'] = data['neg_verb'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['neg_noun'] = {key: val.to(self.device) for key, val in data['neg_noun'].items()}
                data['neg_verb'] = {key: val.to(self.device) for key, val in data['neg_verb'].items()}
            
            data['video'] = data['video'].to(self.device)
            
            # print(data['text']['input_ids'].shape, data['text']['attention_mask'].shape)
            # torch.Size([64, 17]) torch.Size([64, 17])
            
            text_embeds, video_embeds_pooled, negative_noun_embeds, negative_verb_embeds = self.model(data)
            # print(text_embeds.shape, video_embeds_pooled.shape)
            # torch.Size([64, 512]) torch.Size([32, 64, 512])
            
            correct_pair_output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            loss = 0.5*self.loss(correct_pair_output, self.model.clip.logit_scale)
            print(loss)
            text_embeds = text_embeds.norm(dim=-1, keepdim=True)
            negative_noun_embeds = negative_noun_embeds.norm(dim=-1, keepdim=True)
            negative_verb_embeds = negative_verb_embeds.norm(dim=-1, keepdim=True)
            
            dot_product1 = torch.einsum('ij,ij->i', text_embeds, negative_noun_embeds)
            norm_sim = ( dot_product1 + 1 ) / 2
            norm_sim_margin = norm_sim[norm_sim > 0.5]
            loss += 0.25*torch.mean(norm_sim_margin)
            
            dot_product2 = torch.einsum('ij,ij->i', text_embeds, negative_verb_embeds)
            norm_sim = ( dot_product2 + 1 ) / 2
            norm_sim_margin = norm_sim[norm_sim > 0.5]
            loss += 0.25*torch.mean(norm_sim_margin)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res


    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                truncation=True)
                    data['neg_noun'] = self.tokenizer(data['neg_noun'], return_tensors='pt', padding=True,
                                                truncation=True)
                    data['neg_verb'] = self.tokenizer(data['neg_verb'], return_tensors='pt', padding=True,
                                                truncation=True)
                                                
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                    data['neg_noun'] = data['neg_noun'].to(self.device)
                    data['neg_verb'] = data['neg_verb'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['neg_noun'] = {key: val.to(self.device) for key, val in data['neg_noun'].items()}
                    data['neg_verb'] = {key: val.to(self.device) for key, val in data['neg_verb'].items()}
                
                data['video'] = data['video'].to(self.device)
                
                text_embed, vid_embed, vid_embed_pooled, negative_verb_embed, negative_verb_embed = self.model(data, return_all_frames=True)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                
                correct_pair_output = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type)
                total_val_loss = 0.5*self.loss(correct_pair_output, self.model.clip.logit_scale)
                
                text_embed = text_embed.norm(dim=-1, keepdim=True)
                negative_verb_embed = negative_verb_embed.norm(dim=-1, keepdim=True)
                negative_verb_embed = negative_verb_embed.norm(dim=-1, keepdim=True)
                
                dot_product1 = torch.einsum('ij,ij->i', text_embed, negative_verb_embed)
                norm_sim = ( dot_product1 + 1 ) / 2
                norm_sim_margin = norm_sim[norm_sim > 0.5]
                total_val_loss += 0.25*torch.mean(norm_sim_margin)
                
                dot_product2 = torch.einsum('ij,ij->i', text_embed, negative_verb_embed)
                norm_sim = ( dot_product2 + 1 ) / 2
                norm_sim_margin = norm_sim[norm_sim > 0.5]
                total_val_loss += 0.25*torch.mean(norm_sim_margin)
                
                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)
                
            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
             
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            self.model.pool_frames.cuda("cuda:7")

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, 
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)
            
            sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res
        
"""
# TripletMarginLoss Start
# tiling positive pooled video embedding
anchors = video_embeds_pooled[0][0]
anchors = torch.unsqueeze(video_embeds_pooled[0][0], 0)
# print(anchors.shape)    torch.Size([1, 512])
for i in range(1, len(video_embeds_pooled)):
    anchors = torch.cat((anchors, torch.unsqueeze(video_embeds_pooled[i][i], 0)), 0)
# print(anchors.shape)  torch.Size([32, 512])
anchors = anchors.tile((self.batch_size, 1))
# print(anchors.shape)    torch.Size([1024, 512])
pos = text_embeds.tile((self.batch_size, 1))
# print(pos.shape)    torch.Size([1024, 512])
# repeating positives that would act as negatives
neg = text_embeds.repeat_interleave(self.batch_size, dim=0)
# print(neg.shape)    torch.Size([1024, 512])
# change those negative functioning positives which make incorrect pairs ....occur at index 0, 32+1, 64+2,...
# replace them with negatives caption form language model
for j in range(0, len(neg)):
    if (j%(self.batch_size+1) == 0):
        neg[j] = negative_text_embeds[(j)%self.batch_size]

# print(neg.shape)    torch.Size([1024, 512])
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity())

loss = triplet_loss(anchors, pos, neg)
# TripletMarginLoss End

# Cross-EntropyLoss with hardMining
# print(anchors.shape, neg.shape,video_embeds_pooled.shape)

# loss between positive pooled video and positive text
correct_pair_output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
loss = self.loss(correct_pair_output, self.model.clip.logit_scale)

# loss between positive pooled video and negative text
incorrect_pair_output = sim_matrix_training(negative_text_embeds, video_embeds_pooled, self.pooling_type)
sim = torch.diag(incorrect_pair_output)
sim = 1-sim
sim = sim.mul(0.5)
# print(sim)
loss += max(0.3-torch.max(sim), 0.0)
"""