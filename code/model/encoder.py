import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from fsl_project.utils import utils as generic_utils

from utils import utils as generic_utils
from transformers import *
import numpy as np
import warnings
import ast

class Encoder(nn.Module):
	"""
			Multi-head attention encoder
	"""
	def __init__(self, config, embedding,input_channels=1):
		super(Encoder, self).__init__()

		# Generic parameters
		self.args = config
		emb_dim = 768
		self.segment_level = config['seg_level']
		bert_config = BertConfig.from_pretrained('bert-base-uncased')
		self.embed = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
		self.embed.eval()
		self.hidden_size=config['hidden_size']
		self.seg_level = config['seg_level']
		self.max_seg = config['max_seg']
			
	#Segment input: bsz, # seg, max_len
	#Input contains [CLS] and[SEP] tokens	
	def forward(self,input_feat, len_feat, mask,mapping, segmented_index=None, max_segment=None):
		warnings.filterwarnings('ignore')	

		self.embed.eval()
		ori_emb = self.embed(input_ids = input_feat, attention_mask=mask)[0]	
		subtoken_emb = ori_emb[1:-1]
		
		sentence_embedding = ori_emb[:,0] #[CLS] representation
		if (self.args['use_seg'] == 'true'):
			print ("Segmented index", segmented_index)
			segment_outp, segment_masks, seg_scores, seg_full_outp, seg_full_mask = segment_original_single_level(self.args, ori_emb,segmented_index, self.segment_level, self.max_seg)
		else:
			segment_outp, segment_masks, seg_scores, seg_full_outp, seg_full_mask = None,None,None,None,None
		
		return ori_emb,None,sentence_embedding, None,None,None,segment_outp,segment_masks, seg_scores,seg_full_outp, seg_full_mask


def print_test(inp, mask, seg):
	print ("Input", inp[-1])
	print("mask", mask[-1])
	if (seg != None):
		print("segid", seg[-1])
	print ("\n")

#---- Find the maximum length across all segment levels -----#
# --- Return: [#seg] # (i.e. [2,4,6] => seg 1: max_num_seg is 2, seg 2: max_num_seg is 4
#		segmented_input: [bsz, #extracted_seg_levels]
def find_max_num_segment_for_each_level(segmented_input, seg_level,use_rl):
	max_seg= []
	max_seg_len_all=[]
	for i in range(len(seg_level)):
		cur_level = seg_level[i]
		current_input = segmented_input[:,i]
		max_segment = 0	
		test_count = 0
		max_seg_len = 0
		for seg_i in current_input:
			seg_i = ast.literal_eval(seg_i)[0]
			if (cur_level == 0):
				seg_i = [seg_i]
			test_count = 0
			for j in seg_i:
				test_count += 1
				max_seg_len = max(max_seg_len, j[1]-j[0]+1)
			max_segment = max(max_segment,test_count)
		max_seg.append(max_segment)
		max_seg_len_all.append(max_seg_len)


	return max_seg, max_seg_len_all


#input: idx of each token
# segmented_input: list of range for each segment on the same level
# seg_level: 
# max_segment: upper bound #segment on current level
def obtain_current_seg_input(config, input_emb, segmented_input,seg_level, max_segment, use_rl, max_seg_len):
	#print ("Input emb", input_emb.isnan().any())
        counter = 0
        new_input = [] 
        new_full_input = [] # bsz,max_num_seg, max_seg_len, D
        new_full_rand_input=[]
        #Go through each sentence
        scores = []
        max_num_score = max_segment - 1
        seg_mask = torch.zeros(input_emb.shape[0], max_segment) # bsz, #max_num-seg
        rand_seg_mask = torch.zeros(input_emb.shape[0], max_segment) # bsz, #max_num-seg
        seg_mask, rand_seg_mask = generic_utils.move_to_cuda(config, seg_mask, rand_seg_mask)
        element_counter = 0

        for i in segmented_input: # each sample	
                split_score =ast.literal_eval(i[0])[1]
                if (split_score == []):
                        tensor_score = torch.Tensor([-1])
                else:	
                        tensor_score = torch.Tensor(split_score)
                tensor_score = tensor_score.unsqueeze(0)
                if (tensor_score.shape[1] < max_num_score):	
                        padded = torch.zeros(1, max_num_score - tensor_score.shape[1])
                        tensor_score = torch.cat([tensor_score, padded],1)	
                tensor_score = generic_utils.move_to_cuda(config, tensor_score)[0]
                scores.append(tensor_score)	

                i =ast.literal_eval(i[0])[0]
        
                if (seg_level == 0 or type(i) == tuple):
                        i = [i]
                        print ("Should not be here")

		# Equal # segs as the source?
                num_breaks = len(i)
                pos_breaks = np.arange(1,i[-1][-1])
                np.random.shuffle(pos_breaks)
                rand_split = pos_breaks[:num_breaks]
                rand_split = np.sort(rand_split).tolist()
                start= 1
                segs = []
                for idx in range (len(rand_split)):
                        cut_idx = rand_split[idx]
                        cur_point = (start, cut_idx)
                        if (idx == len(rand_split) -1):
                                cur_point = (start, i[-1][-1])
                        segs.append(cur_point)
                        start = cut_idx + 1

                seg_rep, seg_full_rep,seg_mask = generic_utils.convert_tuple_list_to_input(config, i, counter, element_counter, input_emb, seg_mask,max_segment, max_seg_len)

                rand_seg_rep, rand_seg_full_rep,rand_seg_mask = generic_utils.convert_tuple_list_to_input(config, segs, counter, element_counter, input_emb, rand_seg_mask,max_segment, max_seg_len)

                new_full_rand_input.append(rand_seg_full_rep)
                new_input.append(seg_rep)
                new_full_input.append(seg_full_rep)
                counter += 1
                                
        new_input = torch.cat(new_input, 0)
        new_full_input = torch.cat(new_full_input, 0) #bsz,max_num_seg, max_seg_len,D
        new_full_rand_input = torch.cat(new_full_rand_input, 0)
        sum_input = new_full_input.sum(dim=-1)
        new_full_masks = torch.where(sum_input != 0.0, 1.0, 0.0) #bsz, max_num_seg, max_seg_len

        sum_rand_input = new_full_rand_input.sum(dim=-1)
        new_rand_full_masks = torch.where(sum_rand_input != 0.0, 1.0, 0.0) #bsz, max_num_seg, max_seg_len

        #Sanity check
        full_in = new_full_input.sum(dim=-1) #bsz, max_num_seg, max_seg_len
        num_seg = torch.count_nonzero(new_full_masks.sum(dim=-1),dim=1)
        seg_len = torch.count_nonzero(new_full_masks,dim=-1)
        seg_in = torch.count_nonzero(full_in.sum(dim=-1),dim=1)
        seg_len_in = torch.count_nonzero(full_in, dim=-1)
        assert torch.allclose(num_seg, seg_in)
        assert torch.allclose(seg_len_in, seg_len)
        scores = torch.cat(scores,0)
        return new_input,seg_mask, scores, new_full_input, new_full_masks,new_full_rand_input, new_rand_full_masks


def create_token_level(embedding, mapping):
        new_embedding =torch.zeros(embedding.shape)
        if (torch.cuda.is_available()):
                new_embedding = new_embedding.cuda()
        for i in range (embedding.shape[0]):
                cur_emb = embedding[i]
                cur_map = mapping[i]
                for index in range(len(cur_map)):
                        map_e = cur_map[index]
                        if (len(map_e) <=1): # only single word/ special token)
                                new_embedding[i,index] = embedding[i,index]
                        else:
                                mean_pool = embedding[i,map_e[0]:map_e[-1] +1,:].mean(dim=0,keepdim=True)
                                new_embedding[i, index] = mean_pool
        return new_embedding


def segment_original_single_level(config, input_seg,segmented_input,seg_level,max_segment=-1, max_segment_len=-1, use_rl='false'):
	seg_outp = []

	seg_mask = []
	seg_scores=[]
	seg_full_output=[]
	seg_full_mask=[]

	rand_seg_full_output=[]
	rand_seg_full_mask=[]
	max_segment_for_all, max_segment_len_for_all = [max_segment], [max_segment_len] #use max_segment info

	
	max_seg_cur_level = max_segment_for_all[0]
	cur_seg_input,cur_seg_mask, seg_score, cur_full_seg_input, cur_full_seg_mask,cur_rand_full_seg_input, cur_rand_full_seg_mask = obtain_current_seg_input(config,input_seg, segmented_input,seg_level[0], max_seg_cur_level,use_rl, max_segment_len_for_all[0]) # [bsz, max_num_seg, D]
	
	#cur_seg_mask: [bsz,max_seg]
	seg_outp.append(cur_seg_input)
	seg_mask.append(cur_seg_mask)
	seg_scores.append(seg_score)
	seg_full_output.append(cur_full_seg_input)
	seg_full_mask.append(cur_full_seg_mask)

	rand_seg_full_output.append(cur_rand_full_seg_input)
	rand_seg_full_mask.append(cur_rand_full_seg_mask)
	#Fill up the segments
	#print_test(new_input,new_mask,None)
	return seg_outp, seg_mask, seg_scores, seg_full_output, seg_full_mask, rand_seg_full_output, rand_seg_full_mask

def segment_original(config,input_seg,segmented_input,seg_level,max_segment, use_rl):
	seg_outp = []

	seg_mask = []
	seg_scores=[]
	seg_full_output=[]
	seg_full_mask=[]
	if (max_segment != None):
		seg_level = np.arange(max_segment)
	 # ------- Find the maximum number of segments ----------------#
	max_segment_for_all, max_segment_len_for_all = find_max_num_segment_for_each_level(segmented_input, seg_level, use_rl)
	
	for i in range(len(seg_level)):
		cur_seg = seg_level[i]
		seg_input = segmented_input[:,i] # Get the corresponding segment (i.e. [(1,15),[]]) 
		max_seg_cur_level = max_segment_for_all[i]
		cur_seg_input,cur_seg_mask, seg_score, cur_full_seg_input, cur_full_seg_mask  = obtain_current_seg_input(config, input_seg, seg_input,cur_seg, max_seg_cur_level,use_rl, max_segment_len_for_all[i]) # [bsz, max_num_seg, D]
		seg_outp.append(cur_seg_input)
		seg_mask.append(cur_seg_mask)
		seg_scores.append(seg_score)
		seg_full_output.append(cur_full_seg_input)
		seg_full_mask.append(cur_full_seg_mask)
	#Fill up the segments
	#print_test(new_input,new_mask,None)
	return seg_outp, seg_mask, seg_scores, seg_full_output, seg_full_mask
