import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import pass_utils, cl_utils
from utils import utils as generic_utils

from model import pm as perturbed_masking
import copy
import math

def generate_return_dict(tie_break,seg_text, seg_len, seg_len_std, sim_score=None):

	return_dict={}
	return_dict['tie_break'] = tie_break
	return_dict['seg_text'] = seg_text
	return_dict['seg_len'] = seg_len
	return_dict['seg_len_std'] = seg_len_std
	if (sim_score != None):
		return_dict['inter_sim'] = sim_score
	return return_dict
def masked_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec-max_vec)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	zeros=(masked_sums == 0)
	masked_sums += zeros.float()
	return masked_exps/masked_sums

def init_weights(m):
	if (type(m) == nn.Linear):
		torch.nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0.01)

class CL_Model(nn.Module):

	def __init__(self, config):
		super(CL_Model, self).__init__()
		
		self.args = config
		self.seg_level = config['seg_level']

		self.pm = perturbed_masking.Perturbed_Masking(config)
		if (self.args['use_attn_reward'] == 'true'):
			filter_sizes=[1,2,3]
			num_filters = [10,10,10]
			self.conv1d = nn.ModuleList([
				nn.Conv1d(in_channels = config['max_seg'],
					out_channels=num_filters[i],
					kernel_size=filter_sizes[i])
				for i in range(len(filter_sizes))
				])

		else:
			self.conv1d=None

		self.relu_act = nn.ReLU()
		self.sigmoid_act = nn.Sigmoid()

	#Input: [bsz,#seg,#word_in_seg, D]
	def random_mask_word_in_seg(self,input_segment, input_num_seg, input_seg_len, num_seg=-1, ratio_seg=-1):
		output = []
		for element_counter in range (input_segment.shape[0]):
			current_input_segment = input_segment[element_counter] #[max_seg, max_len,D]
			num_total_seg = input_num_seg[element_counter].item() #int 
			seg_len = input_seg_len[element_counter] #[max_seg,1]

			cl_utils.sanity_check_num_seg(current_input_segment, num_total_seg, seg_len, self.args['level_aug'])
			selected_seg_idx = cl_utils.generate_num_mask_seg_word(self.args, ratio_seg, num_seg, num_total_seg) #[num_selected_seg,] (If level=word, either use all or randomly choosing seg first)
			sample = copy.deepcopy(current_input_segment.detach()) #bsz,num_seg, num_word_in_seg, D]

			if (self.args['level_aug'] == 'seg'):
				sample,updated_total_seg = cl_utils.mask_seg_level(self.args['mask_word_side'], selected_seg_idx, sample, seg_len,num_total_seg)	
				input_num_seg[element_counter] = updated_total_seg
				num_total_seg = updated_total_seg #update for sanity check
			else:
				sample, seg_len = cl_utils.mask_word_level(self.args,selected_seg_idx, sample, seg_len, num_total_seg)
			new_seg_len = seg_len.unsqueeze(0) 
			#sample:[1,max_seg,max_len,D], new_seg_len: [1,#max_seg,D]
			#cl_utils.sanity_check_num_seg(sample, num_total_seg, new_seg_len, self.level_aug)

			sample = sample.sum(dim=-2)/ (new_seg_len + 1e-20) #mean over the remaining words in the segment to find segment rep [bsz, num_seg,D]
			assert sample.isnan().any() == False
			output.append(sample)

		output = torch.cat(output,0)
		output.requires_grad=True
		assert output.isnan().any() == False
		return output			


	def transform_sim(self,sim):
		sim = self.relu_act(sim)
		#sim = self.sigmoid_act(sim)
		return sim

	def forward(self, anchor, pos, neg):
		""" forward pass
		Inputs:
				inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
				lengths: lengths of x (bsz, )
				pretrained_seg: (bsz,) : each is the corresponding segment
		Output:
				prediction: Intent prediction (bsz, num_intent)
		"""

		return_anchor, return_pos, return_neg, ori_train_info, pos_train_info, neg_train_info = self.pm(anchor, pos, neg)	
		anchor_emb, anchor_word_mask, anchor_seg_mask, rand_anchor_emb, rand_anchor_mask = pass_utils.extract_full_outp(ori_train_info) #bsz, num_seg, max_seg_len, D -- bsz,#seg,seg_len -- bsz,#seg
		pos_emb, pos_word_mask, pos_seg_mask,_,_ = pass_utils.extract_full_outp(pos_train_info)
		neg_emb, neg_word_mask, neg_seg_mask,_,_ = pass_utils.extract_full_outp(neg_train_info)
		#Assume only 1 level is provided
		anchor_emb, anchor_word_mask, anchor_seg_mask = anchor_emb[0], anchor_word_mask[0], anchor_seg_mask[0]
		ori_pos_emb, pos_word_mask, pos_seg_mask = pos_emb[0], pos_word_mask[0], pos_seg_mask[0]
		ori_neg_emb, neg_word_mask, neg_seg_mask = neg_emb[0], neg_word_mask[0], neg_seg_mask[0]

		ori_anchor_emb = anchor_emb
	#	assert pos_emb.isnan().any() == False
	#	assert pos_seg_mask.isnan().any() == False
	#	assert neg_emb.isnan().any() == False
	#	assert neg_seg_mask.isnan().any() == False
		pos_emb = ori_pos_emb.sum(dim=-2) / (pos_word_mask.sum(dim=-1,keepdim=True)+ 1e-20) #can be 0 due to padded segment
		neg_emb = ori_neg_emb.sum(dim=-2) / (neg_word_mask.sum(dim=-1,keepdim=True) + 1e-20)
			
#--------------------------------------------------------------------
		if (self.args['augment'] == 'true'):
			anchor_num_seg = torch.count_nonzero(anchor_word_mask.sum(dim=-1),dim=1)
			anchor_seg_len = anchor_word_mask.sum(dim=-1, keepdim=True) #bsz,#seg, 1
			if  (self.args['mask_type']=='no_mask'):	
				#anchor_emb: [bsz, max_seg,max_len,D], 
				# anchor_seg_len: [bsz,max_seg,1]
				aug_anchor = anchor_emb.sum(dim=-2) / (anchor_seg_len + 1e-20)
			
			elif (self.args['mask_type']=='mask_seg'):
				if (self.args['fixed_seg'] == 'false'):
					aug_anchor = self.random_mask_word_in_seg(anchor_emb, anchor_num_seg, anchor_seg_len,ratio_seg=self.args['ratio_seg'])
				else:		
					aug_anchor = self.random_mask_word_in_seg(anchor_emb, anchor_num_seg, anchor_seg_len, num_seg=self.args['num_seg'])	
			aug_anchor_mask, aug_anchor_len = cl_utils.create_mask_seg(aug_anchor)

			anchor_emb, anchor_mask = aug_anchor, aug_anchor_mask
			pos_sim, inter_pos_sim,_ = generic_utils.compute_pos_neg_similarity(anchor_emb, pos_emb, anchor_mask, pos_seg_mask, self.transform_sim, self.conv1d, self.args['use_relu'], self.args['naive_meanpool'])

			return_pos['inter_sim'] = inter_pos_sim.cpu().detach().numpy()
			neg_sim, inter_neg_sim,_ = generic_utils.compute_pos_neg_similarity(anchor_emb, neg_emb, anchor_mask, neg_seg_mask, self.transform_sim, self.conv1d, self.args['use_relu'], self.args['naive_meanpool'])

			return_neg['inter_sim'] = inter_neg_sim.cpu().detach().numpy()			
			label_logits = 0.0

			sim_score = torch.cat([pos_sim, neg_sim],1)	
			labels = torch.zeros(sim_score.shape[0], dtype=torch.long) # positive (first one) is the label 
			labels = generic_utils.move_to_cuda(self.args, labels)[0]
			
			logits = sim_score / self.args['sent_temp'] #temperature
		else:
			logits =0.0
			labels = 0.0
			labels=0.0
			label_logits = 0.0

		if (self.args['self_supervise'] == 'true'):
			rand_anchor_emb, rand_anchor_mask = rand_anchor_emb[0], rand_anchor_mask[0] #emb: [bsz,max_seg,max_seg_len,D]
			num_seg = torch.count_nonzero(rand_anchor_mask.sum(dim=-1),dim=1) #[bsz]
			rand_seg_len = rand_anchor_mask.sum(dim=-1, keepdim=True) #[bsz,max_seg, 1]

			anchor_cls = ori_train_info['ori'][:,0:1,:] #[bsz,1,D]	
			rand_emb = rand_anchor_emb.sum(dim=-2) / (rand_seg_len + 1e-20) #[bsz,max_seg,D]
			rand_mask,_ = cl_utils.create_mask_seg(rand_emb) #[bsz,max_seg]
			
			rand_sim_input = rand_emb * rand_mask.unsqueeze(-1)
			src_mask = torch.ones(rand_mask.shape[0],1)
			src_mask = generic_utils.move_to_cuda(self.args, src_mask)[0]

			#---- Traditional Way
			num_act_sim = src_mask * rand_mask.sum(dim=-1,keepdim=True)
			neg_score = F.cosine_similarity(anchor_cls.unsqueeze(2), rand_sim_input.unsqueeze(1), dim=-1)
			neg_score = neg_score.sum(dim=-1).sum(dim=-1, keepdim=True) / (1e-20 + num_act_sim)


			anchor_num_seg = torch.count_nonzero(anchor_word_mask.sum(dim=-1),dim=1)
			anchor_seg_len = anchor_word_mask.sum(dim=-1, keepdim=True) #bsz,#seg, 1
			gold_emb = ori_anchor_emb.sum(dim=-2) / (anchor_seg_len + 1e-20) #[bsz,max_seg,D]
			gold_mask,_ = cl_utils.create_mask_seg(gold_emb)


			gold_sim_input = gold_emb * gold_mask.unsqueeze(-1)
			num_act_sim = src_mask * gold_mask.sum(dim=-1,keepdim=True)
			pos_score = F.cosine_similarity(anchor_cls.unsqueeze(2), gold_sim_input.unsqueeze(1), dim=-1)
			pos_score = pos_score.sum(dim=-1).sum(dim=-1, keepdim=True) / (1e-20 + num_act_sim)

			self_score = torch.cat([pos_score, neg_score],1)

			self_labels = torch.zeros(self_score.shape[0], dtype=torch.long) # positive (first one) is the label 

			self_labels = generic_utils.move_to_cuda(self.args, self_labels)[0]
			self_score = self_score / self.args['seg_temp'] #temperature
		else:
			self_score = 0.0

		return logits,label_logits,labels,self_score,self_labels, return_anchor, return_pos,return_neg			
