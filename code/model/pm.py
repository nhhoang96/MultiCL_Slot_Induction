import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import bert_segment_utils, pass_utils
from utils import utils as generic_utils

from model import encoder
from transformers import *
import copy
import time
import math
SLOT_PAD=0
mask_id = 103

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

class Perturbed_Masking(nn.Module):

	def __init__(self, config):
		super(Perturbed_Masking, self).__init__()
		self.seed = config['cur_seed']
		bert_model_dir ='../bert_model/'
		self.embed = BertModel.from_pretrained(bert_model_dir, output_hidden_states=True)

		self.args = config
		self.test_eff = config['test_eff']

		self.relu_act = nn.ReLU()
		self.sigmoid_act = nn.Sigmoid()
	
	def process_segment(self, embedded_inputs, segmented_idx_dict, level):
		segmented_index = np.array(segmented_idx_dict,dtype='<U2048')
		segment_outp, segment_mask, segment_scores, segment_full_outp, segment_full_mask, segment_rand_full_outp, segment_rand_full_mask = encoder.segment_original_single_level(self.args, embedded_inputs, segmented_index, self.args['seg_level'], self.args['max_seg'], self.args['max_seg_len'],'false')
		return segment_outp, segment_mask, segment_full_outp, segment_full_mask, segment_scores, segment_rand_full_outp, segment_rand_full_mask

	def process_pos_neg_seg(self,sample):	
		sample_inputs,sample_masks, sample_text, sample_y, sample_slot, sample_length, sample_mapping, sample_old_map, sample_tokenized_text, sample_focused_tie_break, sample_full_tie_break, sample_subword = generic_utils.extract_info(sample)
		sample_inputs, sample_masks = generic_utils.convert_full_np_to_long_tensor(self.args, sample_inputs, sample_masks)

		assert sample_inputs.isnan().any() == False
		assert sample_masks.isnan().any() == False
		sample_full_emb, sample_cum_tie_break, sample_segmented_idx_dict, sample_segment_graph,sample_desired_lev = self.build_segment_idx(sample_inputs, sample_masks, sample_old_map, sample_mapping, sample_tokenized_text, sample_subword, self.args['seg_level'], 'sample')
		assert sample_full_emb.isnan().any() == False
		sample_segment_outp, sample_segment_mask, sample_segment_full_outp, sample_segment_full_mask, sample_segment_scores,_,_ = self.process_segment(sample_full_emb, sample_segmented_idx_dict, self.args['seg_level'][0])

		assert sample_segment_full_outp[0].isnan().any() == False
		assert sample_segment_full_mask[0].isnan().any() == False
		return sample_segment_outp, sample_segment_mask, sample_segment_full_outp, sample_segment_full_mask, sample_segment_scores, sample_segmented_idx_dict, sample_cum_tie_break,sample_desired_lev,sample_full_emb

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
	
		inputs,masks, text, y, slot, length, mapping, old_map, tokenized_text, focused_tie_break, full_tie_break, subword = generic_utils.extract_info(anchor)
		inputs, masks = generic_utils.convert_full_np_to_long_tensor(self.args, inputs, masks)

		full_ori_emb, cum_tie_break, segmented_idx_dict,segment_graph,anchor_desired_lev = self.build_segment_idx(inputs, masks, old_map, mapping, tokenized_text, subword, self.args['seg_level'],'original')
		if (self.args['test_mode'] == 'true'):
			print ("Segmented dict", segmented_idx_dict)
		
		segment_outp, segment_mask, segment_full_outp, segment_full_mask, segment_scores, segment_rand_full_outp, segment_rand_full_mask = self.process_segment(full_ori_emb, segmented_idx_dict, self.args['seg_level'][0])
		ori_segment_text,ori_seg_len,ori_seg_len_std = generic_utils.extract_segment_text(anchor['tok_text'], segmented_idx_dict, anchor['mapping'])
		ori_train_info = pass_utils.save_train_info(full_ori_emb, segment_outp, segment_mask, segment_full_outp, segment_full_mask, segment_graph, segment_scores, segmented_idx_dict, anchor_desired_lev,segment_rand_full_outp, segment_rand_full_mask)

		return_ori =generate_return_dict(cum_tie_break, ori_segment_text, ori_seg_len, ori_seg_len_std) #save info 
		
		if (type(pos['x']) != type(None)): # if pos/neg samples are provided
			pos_segment_outp, pos_segment_mask, pos_segment_full_outp, pos_segment_full_mask, pos_segment_scores, pos_segmented_idx_dict,pos_cum_tie_break,_,pos_full_emb = self.process_pos_neg_seg(pos)
			neg_segment_outp, neg_segment_mask, neg_segment_full_outp, neg_segment_full_mask, neg_segment_scores,neg_segmented_idx_dict, neg_cum_tie_break,_, neg_full_emb = self.process_pos_neg_seg(neg)

			pos_train_info = pass_utils.save_train_info(pos_full_emb, pos_segment_outp, pos_segment_mask, pos_segment_full_outp, pos_segment_full_mask, None, pos_segment_scores, pos_segmented_idx_dict, None)
			neg_train_info = pass_utils.save_train_info(neg_full_emb, neg_segment_outp, neg_segment_mask, neg_segment_full_outp, neg_segment_full_mask, None, neg_segment_scores, neg_segmented_idx_dict, None)

			pos_segment_text,pos_seg_len, pos_seg_len_std = generic_utils.extract_segment_text(pos['tok_text'], pos_segmented_idx_dict, pos['mapping'])	
			#pos_sim, inter_pos_sim,prior_pos_sim = generic_utils.compute_pos_neg_similarity(new_original, new_pos, new_mask_original, new_mask_pos, self.transform_sim)
			#neg_segment_text, neg_seg_len,neg_seg_len_std = generic_utils.extract_segment_text(neg['tok_text'], neg_segmented_idx_dict[self.args['seg_level'][0]], neg['mapping'])
			neg_segment_text, neg_seg_len,neg_seg_len_std = generic_utils.extract_segment_text(neg['tok_text'], neg_segmented_idx_dict, neg['mapping'])

			return_pos =generate_return_dict(pos_cum_tie_break, pos_segment_text, pos_seg_len, pos_seg_len_std)
			return_neg =generate_return_dict(neg_cum_tie_break, neg_segment_text, neg_seg_len, neg_seg_len_std)
		else:
			return_pos, return_neg, pos_train_info, neg_train_info = None, None, None, None
		
		return return_ori, return_pos, return_neg, ori_train_info, pos_train_info, neg_train_info

	def build_segment_idx(self,inputs, masks, old_map, mapping, tokenized_text, subword, level, d_type):	
		cum_tie_break = []
		segmented_index=[]
		segment_graph =[]
		full_seg_used=[]
		segment_idx_dict=[]
		desired=[]
		assert inputs.isnan().any() == False
		assert masks.isnan().any() == False

		start_time=time.time()

		torch.manual_seed(self.seed)
		full_emb,_,all_hidden = self.embed(inputs, masks, return_dict=False) #[bsz,max_len,D]			
			
		assert full_emb.isnan().any() == False
		for index in range (inputs.shape[0]):
			indexed_tokens = inputs[index,: masks[index].sum()]
			old_mapping = old_map[index] #[-1, 0, 1,2, 2,... -1]
			subwords = subword[index]
			h_time = time.time()

			self.embed.eval()
			tokens = []
			segments=[]
			new_time=time.time()
			build_t = time.time()	
			#if (torch.cuda.is_available()):
			#	generic_utils.show_gpu("Before loop")

			with torch.no_grad():
				for i in range (0, len(tokenized_text[index])):
					tokens_tensor, segments_tensor = bert_segment_utils.prepare_tensor(self.args, old_mapping, indexed_tokens, tokenized_text[index], i) #[tokenized_len, tokenized_len]			
					tokens.append(tokens_tensor)
					segments.append(segments_tensor)
				tok_tensor = torch.cat(tokens,0)
				seg_tensor = torch.cat(segments,0)
				
				idx_tensor = torch.arange(0,len(tokenized_text[index]))
				idx_tensor = idx_tensor.repeat(len(tokenized_text[index]),) #[max_len **2,]

				emb_t = time.time()
				
				#if (torch.cuda.is_available()):
				#	generic_utils.show_gpu("Before embedding")

				torch.manual_seed(self.seed)
				new_hidden_state = self.embed(tok_tensor, seg_tensor, return_dict=False)[0] #[max_len**2, max_len,D]

				#if (torch.cuda.is_available()):
				#	generic_utils.show_gpu("After embedding")
				new_matrix = new_hidden_state[range(new_hidden_state.shape[0]),idx_tensor,:]	


				new_matrix = new_matrix.view(len(tokenized_text[index]), len(tokenized_text[index]), -1).contiguous()
				new_matrix = new_matrix.transpose(0,1).contiguous() #[max_len, max_len, D]
				np_new_matrix = new_matrix.detach().cpu().numpy() #[maxlen**2, max_len**2, D]
				
				if (torch.cuda.is_available()):
					torch.cuda.empty_cache()
				del new_matrix
			
			self.embed.train()
			indexed_tokens = indexed_tokens.detach().cpu().numpy()
			id_dict, num_occ_dict = build_id_dict(indexed_tokens)
			#0.07% of run-time
			init_matrix, simplified_toks = obtain_impact_matrix(indexed_tokens, np_new_matrix, tokenized_text[index], subwords, old_mapping)	
			seg_time = time.time()

			#Only 0.013% of the run-time
			new_tree,score_split = bert_segment_utils.mart(init_matrix, simplified_toks, self.args['new_score'])

			if (isinstance(new_tree, int)):
				new_tree = [new_tree]

			if not (isinstance(level,list)):
				level =[level]
			tree_time = time.time()
			return_level, tree_depth,desired_level = bert_segment_utils.create_level_dict(new_tree, self.args['seg_level'], id_dict, subwords, old_map[index],tokenized_text[index],indexed_tokens, init_matrix, num_occ_dict, self.args['new_score'], level[0])

			#Assume only 1 level is provided
			return_val = return_level[desired_level]	
			seg = np.array([str(return_val)])
			
			segment_idx_dict.append(seg)
			# Obtain: [1,10]: (2.5) (Assume single level)
			if (len(level) == 1): # only 1 extracted level => generate B-T tags, otherwise: use RL
				return_v = return_val[0] #Ignore the score (only take the seg [(1,2), (3,10)]
				if (desired_level == 0):
					return_v = [return_v]
				tie_break = convert_segment_to_tie(return_v, old_mapping)
			total_distinct_word, total_distinct_count = np.unique(np.array(old_mapping),return_counts=True)
			num_label = len(old_mapping) - (total_distinct_count.sum()- 1* len(total_distinct_count)) - 1 - 1 
			assert len(tie_break) == num_label
			cum_tie_break.append(tie_break)

		segment_graph = None
		return full_emb, cum_tie_break, segment_idx_dict,segment_graph, desired


#------ Return dict
def build_id_dict(indexed_tokens):
	id_dict={}
	num_occ_dict={}
	for i in range (len(indexed_tokens)):
		if not (indexed_tokens[i] in id_dict):
			id_dict[indexed_tokens[i]] = [i]
			num_occ_dict[indexed_tokens[i]] = 1
		else:
			temp = id_dict[indexed_tokens[i]]
			if (isinstance(temp, list)):
				temp.append(i)
			else:
				temp = [temp]
				temp.append(i)
			id_dict[indexed_tokens[i]] = temp

			num_occ_dict[indexed_tokens[i]] += 1
	return id_dict, num_occ_dict

#Matrix: NxN (where cell = [1xD])
def obtain_impact_matrix(indexed_tokens, matrix, tok_text, subwords, mapping):
	init_matrix = np.zeros((len(indexed_tokens), len(indexed_tokens)))
	for i, hidden_states in enumerate(matrix):
		base_state = hidden_states[i]
		for j, state in enumerate(hidden_states):
			init_matrix[i][j] = np.linalg.norm(base_state - state)


	init_matrix,simplified_toks, simplified_text = bert_segment_utils.original_cleanup(init_matrix,mapping, indexed_tokens, tok_text, subwords) 
	init_matrix = init_matrix/ np.linalg.norm(init_matrix, axis=0) #Frobenius Norm (L2 Norm Matrix across rows)

	assert np.isnan(init_matrix).any() == False
	assert np.all(init_matrix==0) == False
	np.fill_diagonal(init_matrix,0.)
	init_matrix = 1. - init_matrix
	np.fill_diagonal(init_matrix,0.)

	return init_matrix, simplified_toks


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0))


def convert_segment_to_tie(return_v, old_mapping):
	tie_break = []
	for idx in range (len(return_v)):
		if (old_mapping[return_v[idx][1]] - old_mapping[return_v[idx][0]] > 0):
			num_tie = old_mapping[return_v[idx][1]] - old_mapping[return_v[idx][0]]
			tie_break.extend([1]*(num_tie))
	
		if (idx != len(return_v) - 1): #add a break between segments	
			tie_break.append(0)

	num_breaks= (old_mapping[-2] + 1) -1 # have 9 elements,  8 breaks
	assert len(tie_break) == num_breaks
		
	return tie_break
