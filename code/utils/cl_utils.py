import numpy as np
import copy
import torch.nn.functional as F
import torch
import math
from utils import utils 

#Input seg: Tensor [bsz, #seg, D]
def create_mask_seg(input_seg):
	sum_input = input_seg.sum(dim=-1)
	mask_tensor = torch.where(sum_input != 0.0, 1.0,0.0)
	len_tensor = (mask_tensor == 1.0).sum(dim=1)
	return mask_tensor, len_tensor


#Default: use only 1 extra positive sample and treat the rest as negative samples
def create_random_compare_idx(dictionary, batch_y, batch_index, num_samples=1):
	new_idx = []
	for idx in range (batch_y.shape[0]):
		item = batch_y[idx]
		idx_list = copy.deepcopy(dictionary[item.item()])
		idx_list = idx_list.reshape(-1,)

		cur_idx = list(idx_list).index(batch_index[idx].item())
		new_idx_list = np.delete(idx_list, cur_idx) # guarantee no repetition

		assert idx_list.shape[0] == new_idx_list.shape[0] + 1

		if (new_idx_list.shape[0] <= num_samples):
			num_samples_needed = num_samples - new_idx_list.shape[0]
			add_list = np.repeat(idx_list,num_samples_needed*2)
			new_idx_list = np.concatenate((new_idx_list, add_list),0)

		np.random.shuffle(new_idx_list)
		if (num_samples >1):
			new_idx.extend(new_idx_list[:num_samples].tolist())
		else:
			new_idx.append(new_idx_list[:num_samples].item())
	new_idx = np.array(new_idx)
	return new_idx


def create_random_neg_idx(dictionary, batch_y, num_samples=1):
	new_idx=[]
	for item in batch_y:
		#Sampling negative within the same negative classes

		all_keys= np.array(list(dictionary.keys()))
		class_options =np.delete(all_keys,np.where(all_keys == item.item()))
		idx_list = np.array([],dtype=np.int64)
		for opt in class_options:
			added = np.array(list(dictionary[opt])).reshape(-1,)
			idx_list = np.concatenate((idx_list, added))

		np.random.shuffle(idx_list)

		if (num_samples >1):
			new_idx.extend(idx_list[:num_samples].tolist())
		else:
			new_idx.append(idx_list[:num_samples].item())
	new_idx = np.array(new_idx)
	return new_idx

def prepare_seg_emb_for_selection(config,x_tr, x_mask,mapping, subwords, tokenized_text, seg_levels, model):
	bs =32
	num_batch = int(math.ceil(x_tr.shape[0] / bs)) 

	all_x_feat=[]
	all_x_mask=[]

	with torch.no_grad():
		for i in range (num_batch):
			begin_index = i * bs
			end_index = min((i + 1) * bs, x_tr.shape[0])
			batch_x, batch_mask = x_tr[begin_index: end_index], x_mask[begin_index:end_index]
			batch_map, batch_subword = mapping[begin_index: end_index], subwords[begin_index:end_index]
			batch_tok_text = tokenized_text[begin_index: end_index]	

			x_tensor, mask_tensor = utils.convert_full_np_to_long_tensor(config, batch_x,batch_mask)
			x_feat,_, segment_idx_dict, _,_ = model.pm.build_segment_idx(x_tensor,mask_tensor, batch_map,None, batch_tok_text, batch_subword,seg_levels,None)

			_,_,x_full_outp, x_full_mask,_,_,_ = model.pm.process_segment(x_feat, segment_idx_dict, seg_levels)
			seg_len = x_full_mask[0].sum(dim=-1, keepdim=True)
			x_out = x_full_outp[0].sum(dim=-2)/ (seg_len + 1e-20)
			new_mask,_ = utils.create_mask_seg(x_out) 
			all_x_feat.append(x_out)
			all_x_mask.append(new_mask)
	all_x_feat = torch.cat(all_x_feat, dim=0)
	all_x_feat = all_x_feat.detach()
	all_x_mask = torch.cat(all_x_mask,dim=0).detach()
	return all_x_feat, all_x_mask


def prepare_emb_for_selection(config,x_tr, x_mask,embed_model):
	bs = 32
	num_batch = int(math.ceil(x_tr.shape[0] / bs)) 

	all_x_feat=[]
	all_x_mask=[]
	for i in range (num_batch):
		begin_index = i * bs
		end_index = min((i + 1) * bs, x_tr.shape[0])
		batch_x, batch_mask = x_tr[begin_index: end_index], x_mask[begin_index:end_index]
		with torch.no_grad():
			x_tensor, mask_tensor = utils.convert_full_np_to_long_tensor(config, batch_x,batch_mask)
			x_feat = embed_model(x_tensor, mask_tensor)[0][:,0:1,:] # all tokens in the sentence
			all_x_feat.append(x_feat)
			all_x_mask.append(mask_tensor)
	all_x_feat = torch.cat(all_x_feat, dim=0)
	all_x_feat = all_x_feat.detach()
	all_x_mask = torch.cat(all_x_mask,dim=0).detach()
	return all_x_feat, all_x_mask


def create_sampling_dict(y_te):
	unique_samples,unique_counts = np.unique(y_te, return_counts=True) #Ex: [1,3,4] : [500, 20, 35]
	dictionary={}
	for label_id in unique_samples:
		indexes = np.argwhere(y_te == label_id)
		dictionary[label_id] = indexes
	for idx in range (len(dictionary)):
		assert len(dictionary[list(dictionary.keys())[idx]]) == unique_counts[idx]

	return dictionary


#----- Masking Utilities -----#
#Input: [#seg, seg_len,D]
def sanity_check_num_seg(input_segment, num_total_seg, seg_len, level_aug):
	partial_sum = input_segment.sum(dim=-1) # [1,#seg, seg_len]
	sanity_seg_len = torch.count_nonzero(partial_sum,dim=-1)
	sanity_num_seg = torch.count_nonzero(partial_sum.sum(dim=-1)).item()
	assert sanity_num_seg == int(num_total_seg)
	assert torch.allclose(sanity_seg_len, seg_len.view(-1,).long())


def find_need_masking(ratio_mask, num_mask, num_total_seg, use_ceil):
    if (ratio_mask != -1):
        if (use_ceil == 'false'):
            need_masking = math.floor(ratio_mask * num_total_seg)
        else:
            need_masking = math.ceil(ratio_mask * num_total_seg)
    else:
        if (num_mask > num_total_seg):
            need_masking = num_total_seg
        else:
            need_masking = num_mask
    return int(need_masking)

def generate_num_mask_seg_word(args, ratio_seg, num_seg, num_total_seg, rand_seg='true'):
    # Seg: subset of segments, Word: all segments
    random_int = np.arange(num_total_seg)
    np.random.shuffle(random_int)
    need_masking = find_need_masking(ratio_seg, num_seg, num_total_seg, args['use_ceil'])

    # Seg: subset of segments. Word: all segments
    if (args['level_aug'] == 'both' or args['level_aug'] == 'seg'):
        num_masked_seg = random_int[:need_masking]
    else:
        if (rand_seg == 'true'):
            num_masked_seg = random_int[:need_masking]
        else:
            num_masked_seg = random_int[:int(num_total_seg)]
    return num_masked_seg



#sample: [1,max_seg,max_len,D], seg_len:[max_seg,1]
def mask_seg_level(word_side,selected_seg_idx, sample, seg_len,seg_num):
	original_num_seg = torch.count_nonzero(sample.sum(dim=-1).sum(dim=-1)).item()
	mod_idx=[]
	for index in range (selected_seg_idx.shape[0]): #selected_seg_idx=[#selected_seg,1]
		cur_index = int(selected_seg_idx[index].item())
		if (word_side == 'itself'):	
			sample[cur_index, 0:int(seg_len[cur_index]),:] = 0.0
			seg_len[cur_index] = 0.0
			if (cur_index not in mod_idx):
				mod_idx.append(cur_index)
			#seg_len
		elif (word_side == 'lr'):
			if (cur_index == 0): #mask only right side of first token
				sample[cur_index+1,0: int(seg_len[cur_index + 1]),:] =0.0 #mask the next seg
				seg_len[cur_index+1] = 0.0
				if (cur_index + 1 not in mod_idx): 
					mod_idx.append(cur_index + 1)
			elif (cur_index == seg_num -1): # mask only left token of the last token
				sample[cur_index-1,0:int(seg_len[cur_index-1]),:] = 0.0
				seg_len[cur_index-1] = 0.0

				if (cur_index - 1 not in mod_idx): 
					mod_idx.append(cur_index - 1)
			else: # Middle: do both
				sample[cur_index-1,0:int(seg_len[cur_index-1]),:] = 0.0
				sample[cur_index+1,0: int(seg_len[cur_index+1]),:] =0.0 #mask first token of the next one

				seg_len[cur_index+1] = 0.0
				seg_len[cur_index-1] = 0.0


				if (cur_index - 1 not in mod_idx): 
					mod_idx.append(cur_index - 1)

				if (cur_index + 1 not in mod_idx): 
					mod_idx.append(cur_index + 1)
	after_mod_seg = torch.count_nonzero(sample.sum(dim=-1).sum(dim=-1)).item()
	assert original_num_seg == after_mod_seg + len(mod_idx)
		
	return sample, after_mod_seg


def mask_word_level(args, selected_seg_idx, sample, seg_len,seg_num):
	for index in range(selected_seg_idx.shape[0]): #for each segment idx
		cur_index = int(selected_seg_idx[index].item())
		if (args['mask_word_side'] == 'all' or args['mask_word_side']=='itself'):
			if (seg_len[cur_index] > 1):
				ratio_mask = args['ratio_seg']

				cur_word_masking = generate_num_mask_seg_word(args, ratio_mask, seg_num,seg_len[cur_index].item(),rand_seg=args['rand_seg'])
				zero_tensors = torch.zeros(size=(len(cur_word_masking),sample.shape[-1]))
				zero_tensors = zero_tensors.to(args['gpu_id'])
				sample[cur_index,cur_word_masking] = zero_tensors
				seg_len[cur_index] = seg_len[cur_index] - len(cur_word_masking)

		#Mask left and right
		#sample: max_seg, max_len,D, seg_len: [max_seg,1]
		if (args['mask_word_side'] == 'all' or args['mask_word_side'] =='lr'):
			ratio_mask = args['ratio_seg']

			cur_word_masking = generate_num_mask_seg_word(args, ratio_mask, seg_num,seg_len[cur_index].item(),rand_seg=args['rand_seg']) #list of indexes

			track_idx=[]
			for cur_word_index in cur_word_masking:
				cur_word_index = int(cur_word_index.item())
				if (cur_word_index == 0): #First seg: mask first token of the next one
					if (cur_word_index + 1 not in track_idx): # it has not been masked before, then seg_len reduce, otherwise, keep as is
						sample[cur_index,cur_word_index+1,:] =0.0 #mask first token of the next one
						track_idx.append(cur_word_index + 1)
						updated_len = torch.count_nonzero(sample[cur_index].sum(dim=-1),dim=-1)
						seg_len[cur_index] = updated_len
					
				elif (cur_word_index == seg_num-1): # Last seg: mask the last token of previous one
					if (cur_word_index - 1 not in track_idx): # it has not been masked before, then seg_len reduce, otherwise, keep as is
						sample[cur_index,cur_word_index-1,:] =0.0 #mask first token of the next one
						track_idx.append(cur_word_index - 1)


						updated_len = torch.count_nonzero(sample[cur_index].sum(dim=-1),dim=-1)
						seg_len[cur_index] = updated_len
				else: # Middle seg: do both
					if (cur_word_index - 1 not in track_idx): # it has not been masked before, then seg_len reduce, otherwise, keep as is

						sample[cur_index,cur_word_index-1,:] =0.0 #mask first token of the next one
						track_idx.append(cur_word_index - 1)

						updated_len = torch.count_nonzero(sample[cur_index].sum(dim=-1),dim=-1)
						seg_len[cur_index] = updated_len

					if (cur_word_index + 1 not in track_idx): # it has not been masked before, then seg_len reduce, otherwise, keep as is
						sample[cur_index,cur_word_index+1,:] =0.0 #mask first token of the next one
						track_idx.append(cur_word_index + 1)

						updated_len = torch.count_nonzero(sample[cur_index].sum(dim=-1),dim=-1)
						seg_len[cur_index] = updated_len
	return sample, seg_len
