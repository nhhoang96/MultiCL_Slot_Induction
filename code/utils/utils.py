# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from seqeval.metrics import precision_score, recall_score, f1_score

from model import conll2002_metrics
from preprocess import load_helper
import torch
import copy
import subprocess
import torch.nn.functional as F
import torch.nn as nn
import math
import ast
import re
import os
import copy
import csv
import shutil

def MakeDirectory(path):
	if (os.path.exists(path)):
		shutil.rmtree(path)
	os.mkdir(path)


def extract_data(index, x,mask,text,y,slot, text_len, mapping, old_subword_mapping,tok_text,focused_tie_break, full_tie_break, subword, label_tok_id, label_tok_mask, label_tok_name, label_subword, label_old_map, dictionary, ignore=False):
        if (ignore == True):
                dictionary = save_info(dictionary, None,None, None, None, None, None, None, None, None,None, None,None, None, None,None,None,None)
        else:	
                return_x = x[index]
                return_mask = mask[index]
                return_text = text[index]
                return_y = y[index]
                return_len = text_len[index]
                return_mapping = mapping[index]
                return_slot = slot[index]
                return_tok_text = tok_text[index]
                return_focused_tie_break = focused_tie_break[index]
                return_full_tie_break = full_tie_break[index]
                return_subword = subword[index]
                return_old_subword_mapping = old_subword_mapping[index]

                return_label_tok_id = label_tok_id[index]
                return_label_tok_mask = label_tok_mask[index]
                return_label_tok_name = label_tok_name[index]
                return_label_subword = label_subword[index]
                return_label_old_map = label_old_map[index]

                dictionary = save_info(dictionary, return_x,return_mask, return_text, return_y, return_slot, return_len, return_mapping, return_old_subword_mapping,return_tok_text, return_focused_tie_break, return_full_tie_break, return_subword, return_label_tok_id, return_label_tok_mask, return_label_tok_name, return_label_subword, return_label_old_map)
        return dictionary


def compute_slot_metrics(slot_preds, out_slot_labels_ids, slot_label_map):
    pad_token_label_id = 0
    out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                    if out_slot_labels_ids[i, j] != pad_token_label_id:
                            out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                            slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    assert len(slot_preds_list) == len(out_slot_label_list)

    results = {}
    slot_result = get_slot_metrics(slot_preds_list, out_slot_label_list)

    results.update(slot_result)
    return results

def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }



def obtain_slot_metric(batch_label, pred, index2slot):
	slot_label = []	
	for a in range (batch_label.shape[0]):
		slot_label.extend(list(batch_label[a]))
	slot_pred = []
	for a in range (len(pred)):
		slot_pred.extend(list(pred[a]))
	label = slot_label
	pred = slot_pred
	lines=[]
	gold_label = list(label)
	prediction = list(pred)

	for pred_index, gold_index in zip(prediction, gold_label): # pred_index: actual length, gold_index: for actual length
		pred_slot = index2slot[pred_index]
		gold_slot = index2slot[gold_index]
		lines.append("w" + " " + pred_slot + " " + gold_slot)
	results = conll2002_metrics.conll2002_measure(lines)
	return results['p'], results['r'], results['fb1']


def convert_slot_to_tie_break(slot_elements, idx2slot=None,slot_set=None):
        focused_tie_break=[]
        full_tie_break = []
        prev_slot=None
        idx_track =0
        for s in slot_elements:
                if (type(s) != int):
                        if (prev_slot != None): # further sequence	
                                if (s == 'O'):
                                        full_tie_break.append(0)
                                        if (prev_slot == 'O'):
                                                focused_tie_break.append(-1)
                                        else:
                                                focused_tie_break.append(0)
                                else:
                                        if (prev_slot == 'O'):
                                                #Remove single slot
                                                focused_tie_break.append(0)
                                                full_tie_break.append(0)
                                        else:
                                                cur_char,type_char = s.split('-')
                                                prev_char, prev_type_char = prev_slot.split('-')
                                                if (cur_char == 'I' and (prev_char == 'B' or prev_char == 'I') and (type_char == prev_type_char)):
                                                        focused_tie_break.append(1)
                                                        full_tie_break.append(1)
                                        
                                                #Remove single slot
                                                else:
                                                        focused_tie_break.append(0)
                                                        full_tie_break.append(0)
                else:
                                focused_tie_break.append(s)
                                full_tie_break.append(s)
                prev_slot = s
                idx_track += 1

        assert len(focused_tie_break) == len(slot_elements)-1
        assert len(full_tie_break) == len(slot_elements) - 1
        return focused_tie_break, full_tie_break


def obtain_tie_break_metric(batch_label, batch_full_label, pred):
	focus_metric={
		'break':{
			'p':[],
			'r':[],
			'f1':[]
			},

		'tie':{
			'p':[],
			'r':[],
			'f1':[]
			}
		}

	full_metric={
		'break':{
			'p':[],
			'r':[],
			'f1':[]
			},

		'tie':{
			'p':[],
			'r':[],
			'f1':[]
			}
		}
	l_metric=['break', 'tie']
	for a in range (batch_label.shape[0]):
		golden_label = np.array(batch_label[a])
		golden_full_label = np.array(batch_full_label[a])
		pred_label = np.array(pred[a])
		assert len(golden_label) == len(pred_label)
		assert len(golden_full_label) == len(pred_label)

		f_p,f_r,f_f1,_ = precision_recall_fscore_support(golden_full_label, pred_label, labels=[0,1], zero_division=0)
		assert len(f_p) ==2 # for break and tie
		for idx in range (len(f_p)):
			l = l_metric[idx] #break/ tie
			full_metric[l]['p'].append(f_p[idx])
			full_metric[l]['r'].append(f_r[idx])
			
		# Trim -1 (non-relevant slot labels)
		focused_idx = np.argwhere(golden_label != -1)
		golden_label = golden_label[focused_idx].reshape(-1,)
		focused_pred_label = pred_label[focused_idx].reshape(-1,)

		# If there is no actual slot phrases to predict,B-T performance will all be 0s
		if (golden_label.tolist() == [-1]* len(golden_label)): #ignore all
			for idx in range (len(l_metric)):
				l = l_metric[idx] #break/ tie
				focus_metric[l]['p'].append(0.0)
				focus_metric[l]['r'].append(0.0)

		else:
		
			p,r,f1,_ = precision_recall_fscore_support(golden_label, focused_pred_label, zero_division=0, labels=[0,1])
			assert len(p) == 2

			for idx in range (len(f_p)):
				l = l_metric[idx] #break/ tie
				focus_metric[l]['p'].append(p[idx])
				focus_metric[l]['r'].append(r[idx])
	return full_metric, focus_metric
	

#--- Show GPU usage for debugging -----#
def show_gpu(msg):
	"""
	ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
	"""
	def query(field):
			return(subprocess.check_output(
					['nvidia-smi', f'--query-gpu={field}',
							'--format=csv,nounits,noheader'], 
					encoding='utf-8'))
	def to_int(result):
			return int(result.strip().split('\n')[0])
	
	used = to_int(query('memory.used'))
	total = to_int(query('memory.total'))
	pct = used/total
	print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')  

#--- Weight initialization
def init_weights(m):
	if (type(m) == nn.Linear):
		torch.nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0.01)
	elif (type(m) == nn.LSTM):
		for param in m.parameters():
			if (len(param.shape) >=2):
				torch.nn.init.orthogonal_(param.data)
			else:
				torch.nn.init.normal_(param)

def load_data(data,eval_type):

		x_te = data['x_'+eval_type]
		y_te = data['y_'+eval_type]
		te_len = data['len_'+eval_type]
		te_mask = data['mask_' + eval_type]
		text_te = data['text_'+eval_type]

		old_subword_mapping_te = data['old_subword_mapping_'+eval_type]
		map_te = data['mapping_'+eval_type]
		subword_te = data['subword_'+eval_type]
		tok_text_te = data['tok_text_'+eval_type]

		label_tok_id_te = data['label_tok_id_'+eval_type]
		label_tok_mask_te = data['label_tok_mask_'+eval_type]
		label_tok_name_te = data['label_tok_name_'+eval_type]
		label_subword_te = data['label_subword_'+eval_type]
		label_old_map_te = data['label_old_subword_mapping_'+eval_type]

		slot_te= data['slot_'+eval_type]
		focused_tie_label_te = data['focused_tie_break_label_'+eval_type]
		full_tie_label_te = data['full_tie_break_label_'+eval_type]

		return x_te, y_te, te_len, te_mask, text_te, map_te,subword_te, old_subword_mapping_te, tok_text_te, subword_te, label_tok_id_te, label_tok_mask_te, label_tok_name_te, label_subword_te, label_old_map_te, slot_te, focused_tie_label_te, full_tie_label_te


# Create one-hot encoded label (0.0 0.0 1.0) from label (2) (example)
def create_index(y):
		sample_num = y.shape[0]
		labels = np.unique(y)
		class_num = labels.shape[0]
		ind = np.zeros((sample_num, class_num),dtype=np.float32)
		labels = range(class_num)
		for i in range(class_num):
				ind[y == labels[i],i] = 1
		return ind


#----- Additional functions -----------#
# Repopulate with examples from the class that did not meet requirements
def repopulate(remain_x, remain_y, remain_y_ind,	remain_len, index, x_tr_ori, y_tr_ori, y_ind_ori, s_len_ori):
		
		new_x_tr = np.concatenate((remain_x,x_tr_ori[index]), axis=0)
		new_y_tr = np.concatenate((remain_y, y_tr_ori[index]), axis=0)
		new_y_ind = np.concatenate((remain_y_ind, y_ind_ori[index]), axis=0)
		new_s_len = np.concatenate((remain_len, s_len_ori[index]), axis=0)

		return new_x_tr, new_y_tr, new_y_ind, new_s_len
				 
def update_index(index, x_tr, y_tr, y_ind, s_len):
		x_tr = np.delete(x_tr, index, axis=0)
		y_tr = np.delete(y_tr, index,axis=0)
		y_ind = np.delete(y_ind, index, axis=0)
		s_len = np.delete(s_len, index, axis=0)
		return x_tr, y_tr, y_ind, s_len

# Ensure that each of n classes has enough n_shot + n_query samples
# index_list: Keep track of the indexes of elements in the current copy
# index_key: Keep track of indexes of elements in the original
def update_with_requirements(cs, index_list, idx_key, required_num, x_tr_copy, y_tr_copy, y_ind_copy, s_len_copy, x_tr, y_tr, y_ind, s_len):
		# Check the requirements
		for idx_class in range(len(cs)):
				if(len(index_list[idx_class]) <= required_num):
						# Obtain indexes of the given class from the original
						index_list[idx_class] = copy.deepcopy(idx_key[cs[idx_class]])
						indexes = np.array(index_list[idx_class])

						x_tr_copy, y_tr_copy, y_ind_copy, s_len_copy = repopulate(x_tr_copy, y_tr_copy, y_ind_copy, s_len_copy, indexes, x_tr, y_tr, y_ind, s_len)
						np.random.shuffle(index_list[idx_class])
						idx_sample = index_list[idx_class][:required_num]
		return index_list, x_tr_copy, y_tr_copy, y_ind_copy, s_len_copy

def upsample(x_te, y_te_id, u_len, batch_size):

		num_classes, count = np.unique(y_te_id, return_counts=True)
		max_class, max_count = num_classes[np.argmax(count)], count[np.argmax(count)]
		max_int_count = int(math.ceil(max_count / batch_size)) * batch_size
		
		all_indices = []
		#Upsample 
		for c in num_classes:
				index_list = np.where(y_te_id == c)[0]
				np.random.shuffle(index_list)
				random_indices = index_list[: max_int_count - count[c]]
				new_indices = np.concatenate((index_list,random_indices), axis=0)
				all_indices.append(new_indices)
		
		all_indices = np.array(all_indices)
		all_indices = all_indices.reshape(-1,)

		x_te_new = x_te[all_indices]
		y_te_new = y_te_id[all_indices]
		u_len_new = u_len[all_indices]

		return x_te_new, y_te_new, u_len_new, max_count

def filter_data(index, query_class, query_attn, query_text):
		new_class= query_class[index]
		new_attn = query_attn[index]
		new_text = query_text[index]
		return new_class, new_attn, new_text


#----Data Loading Helper -----#
def generate_tok_idx(tokens, word2idx):
		tok_idx = []
		for t in tokens:
				if not (t in word2idx):
						tok_idx.append(word2idx['null'])
						#num_not_found_word += 1
				else:
						tok_idx.append(word2idx[t])

		tok_idx = np.array(tok_idx)
		return tok_idx


def create_mask(tokens, max_len, tok_idx):
		if (len(tokens) < max_len):
				tmp = np.append(tok_idx, np.zeros((max_len - len(tokens),), dtype= np.int64))
		else:
				tmp = np.array(tok_idx[0:max_len])
		current_mask = np.ones(shape=tmp.shape)
		for j in range (tmp.shape[0]):
				if (j >= len(tokens)):
						current_mask[j]  = 0.0
		#mask.append(current_mask)
		return tmp, current_mask


def create_subword(tokenized_text, mask):
		current_subword = np.zeros(shape=mask.shape)
		subword_idx = [i for i in range(len(tokenized_text)) if tokenized_text[i].startswith('##')]
		for idx in subword_idx:
				current_subword[idx] = 1.0
		return current_subword

#From -utils ------#
def shuffle_data(x,y,len,mask,text,seg):
		index = np.arange(y.shape[0])
		np.random.shuffle(index)
		#rnd.shuffle(index)
		new_y = y[index]
		new_x = x[index]
		new_len = len[index]
		new_mask = mask[index]
		new_text = text[index]
		new_seg = seg[index]
		return new_x,new_y,new_len,new_mask, new_text,new_seg

#Produce n classes from the total number of classes
def produce_chosen_class(available_classes, y_tr_copy, num_class):
		index_list = []
		chosen_classes = copy.deepcopy(available_classes)
		# Randomly choose n classes from data
		np.random.shuffle(chosen_classes)
		cs = chosen_classes[:num_class]
		#------ Ensure the chosen classes meet the requirements (have at least num_shot + num_query) ---------#
		# Create the updated indexes of the chosen class (this is dynamic from episodes to episodes)
		for val in available_classes:
				index = np.where(y_tr_copy == val)[0]
				index_list.append(index)
		
		return cs, index_list

def create_samples(feature,label,len,ind,mask,text, seg, num_sample, chosen_classes):
		cur_loc = 0
		old_loc = 0
		num_seg = seg.shape[1]
		sub_feature, sub_class, sub_len, sub_ind, sub_mask = init_support_query(num_sample, feature.shape[1], chosen_classes.shape[0])
		sub_text = np.empty(shape=(num_sample * chosen_classes.shape[0],),dtype=text.dtype)
		sub_seg = np.empty(shape=(num_sample * chosen_classes.shape[0],num_seg), dtype='<U2048')

		for cur_class in chosen_classes:

				class_index = np.where(label == cur_class)[0]
				while (int(class_index.shape[0]) < num_sample):
						shuffle_index = copy.deepcopy(class_index)
						np.random.shuffle(shuffle_index)
						class_index= np.concatenate((class_index, shuffle_index[: int(num_sample - class_index.shape[0])]))
				np.random.shuffle(class_index)
				support_index = class_index[:num_sample]
				old_loc = cur_loc
				cur_loc = cur_loc + num_sample
				sub_feature[old_loc: cur_loc,:] = feature[support_index]
				sub_class[old_loc: cur_loc,] = label[support_index]
				sub_len[old_loc:cur_loc,] = len[support_index] 
				sub_mask[old_loc:cur_loc,] = mask[support_index] 
				sub_text[old_loc:cur_loc,] = text[support_index] 
				sub_seg[old_loc:cur_loc,] = seg[support_index] 
		index_key = {}
		
		for i in range (chosen_classes.shape[0]):
				cur_index = np.where(sub_class == chosen_classes[i])[0]
				index_key[chosen_classes[i]] = cur_index
		# Reset class name (0-2-4) => (0-1-2)
		for i in range (chosen_classes.shape[0]):
				sub_class[index_key[chosen_classes[i]]] = i
		sub_ind = create_index(sub_class)
		return sub_feature, sub_class, sub_len, sub_ind, sub_mask, sub_text, sub_seg

#------ Compute Prediction -----#
def compute_prediction(config, prediction):
	#assert config['model_type'] == 'ml'
	if (config['vote_type'] == 'ensemble'):
		if (config['ensemble_type'] == 'max_count'):
			new_preds=[]
			for e in range(prediction.shape[0]):
				#prediction = torch.mean(prediction, 0)
				pred = np.argmax(prediction[e].cpu().detach().numpy(), 1)
				pred = np.expand_dims(pred,0)
				pred = np.transpose(pred)
				new_preds.append(pred)
			
				#unseen_pred = np.argmax(unseen_prediction.cpu().detach().numpy(), 1)
				#seen_pred = np.argmax(seen_prediction.cpu().detach().numpy(), 1)
				#unseen_pred[unseen_pred == 0] = 5
				#unseen_pred[unseen_pred == 1] = 6
			new_preds = np.concatenate(new_preds,1)
			pred = []
			for e in range (new_preds.shape[0]):
				val,count = np.unique(new_preds[e], return_counts=True)
				max_count = np.argmax(count)
				pred.append(val[max_count])
			pred = np.array(pred)
		else:
			pred = np.argmax(prediction.cpu().detach().numpy(), 1)
	else:
		pred = np.argmax(prediction.cpu().detach().numpy(), 1)
	return pred

#_----- Segment Helper ------#
# Segment: [bsz, # seg, L, D]
def pad_segment(segment_rep, max_segment):
	num_seg = segment_rep.shape[1]
	padded_seg = []
	padded_seg.append(segment_rep)
	if (num_seg < max_segment):
		additional_seg = torch.zeros(segment_rep.shape[0], max_segment - num_seg, segment_rep.shape[2])
		if (torch.cuda.is_available()):
			additional_seg = additional_seg.cuda()
		padded_seg.append(additional_seg)
	padded_seg = torch.cat(padded_seg, dim=1)
	return padded_seg

# Simple version: bsz, # seg, D
# Complicated: bsz, #seg, max_len, D
#def reshape_seg(seg, type, num_class, num_support, num_query, batch):
#	# Obtain max_len if needed as dim[-2]
#	# Extend num_seg, len, -1
#	
#	num_segment = seg.shape[1]
#	if (type == 'support'):
#		new_seg = seg.unsqueeze(0).view(num_class, num_support, num_segment, -1).contiguous()
#		new_seg = new_seg.unsqueeze(0).view(1,num_class,num_support,num_segment,-1).contiguous()
#		new_seg = new_seg.unsqueeze(0).view(batch,1, num_class,num_support,num_segment,-1).contiguous()
#		new_seg = new_seg.expand(batch, num_query,num_class,num_support, num_segment,-1).contiguous()
#		new_seg = new_seg.view(batch*num_query*num_class,num_support*num_segment,-1)
#	else:
#		new_seg = seg.unsqueeze(1).view(num_query, 1, num_segment, -1).contiguous() # NQ, 1,len, D
#		new_seg = new_seg.unsqueeze(0).view(batch, num_query,1,num_segment, -1).contiguous()
#		new_seg = new_seg.expand(batch, num_query,num_class,num_segment,-1).contiguous()
#		new_seg = new_seg.view(batch*num_query*num_class,num_segment,-1) #NQN, max_len, D)
#	return new_seg

# a : ax D, b : bx D
def cosine_sim(a, b):
	a_norm = a / a.norm(dim=1)[:,None]
	b_norm = b / b.norm(dim=1)[:,None]
	res = torch.mm(a_norm, b_norm.transpose(0,1))
	return res
#segment: list [#seg, bsz, max_num_segment_for_level, D]
def obtain_segment_rep(segment):
	segment_rep = []
	max_seg = 0
	for s in segment:
		max_seg = max(max_seg, s.shape[1])
	for s in segment:
		pad_s = pad_segment(s, max_seg)
		segment_rep.append(pad_s.unsqueeze(1))
	segment_rep = torch.cat(segment_rep,1)
	return segment_rep

def masked_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec-max_vec)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	zeros=(masked_sums == 0)
	masked_sums += zeros.float()
	return masked_exps/masked_sums

def save_info(sample_dict, x, mask, text, y, slot, length, mapping, old_mapping,tok_text, focused_tie_break, full_tie_break, subword, label_tok_id, label_tok_mask,label_tok_name, label_subword, label_old_mapping):


	sample_dict['x'] = x
	sample_dict['mask'] = mask
	sample_dict['text'] = text
	sample_dict['y'] = y
	sample_dict['slot'] = slot
	sample_dict['len'] = length
	sample_dict['mapping'] = mapping
	sample_dict['old_mapping'] = old_mapping # from Perturbed Masking work
	sample_dict['tok_text'] = tok_text
	sample_dict['focused_tie_label'] = focused_tie_break
	sample_dict['full_tie_label'] = full_tie_break
	sample_dict['subword'] = subword

	sample_dict['label_tok_id'] = label_tok_id
	sample_dict['label_tok_mask'] = label_tok_mask
	sample_dict['label_tok_name'] = label_tok_name
	sample_dict['label_subword'] = label_subword
	
	sample_dict['label_old_mapping'] = label_old_mapping
	return sample_dict

def extract_info(sample_dict):
	return sample_dict['x'] , sample_dict['mask'], sample_dict['text'],sample_dict['y'], sample_dict['slot'], sample_dict['len'], sample_dict['mapping'], sample_dict['old_mapping'], sample_dict['tok_text'], sample_dict['focused_tie_label'], sample_dict['full_tie_label'],sample_dict['subword'] 
	return sample

def extract_subresults(pred, query_class, tokenizer, report_results, index_loc):

	support=[]
	query=[]
	all_q_text = report_results['query']['text']
	for k in list(report_results.keys()):
		for subkey in list(report_results[k].keys()):
			if (subkey == 'feature' or subkey=='graph'):
				report_results[k][subkey] = report_results[k][subkey][index_loc,:]
			else:
				report_results[k][subkey] = report_results[k][subkey][index_loc]

	
	# Sanity check	
	for k in list(report_results.keys()):
		for subkey in list(report_results[k].keys()):
			assert report_results[k][subkey].shape[0] == index_loc.shape[0]

	incorrect_pred, incorrect_class = pred[index_loc], query_class[index_loc]
	query_dict = report_results['query']
	support_dict = report_results['support']
	diff_class_pred = []
	diff_class_true = []
	for i in range (index_loc.shape[0]):
		tokenized_text = tokenizer.convert_ids_to_tokens(query_dict['feature'][i].squeeze()[0].squeeze().tolist())
		print ("Text-Pred-Golden Label:", tokenized_text, query_dict['text'][i], len(tokenized_text), incorrect_pred[i], incorrect_class[i])
		print ("Query Actions", query_dict['action'][i])
		print ("Support Actions", support_dict['action'][i])

		print ("Q-Graph:", query_dict['graph'][i][0,0].nodes.data())

		pred_class = incorrect_pred[i]
		true_class =	incorrect_class[i]

		pred_action = np.array(query_dict['action'][i][0,pred_class].tolist()[0])
		true_s_action = np.array(support_dict['action'][i][0, true_class].tolist()[0])
		pred_s_action = np.array(support_dict['action'][i][0, pred_class].tolist()[0])

		
		diff_true = abs(pred_action[np.where(pred_action == 1)[0]].shape[0] - true_s_action[np.where(true_s_action == 1)[0]].shape[0])
		diff_pred = abs(pred_action[np.where(pred_action == 1)[0]].shape[0] - pred_s_action[np.where(pred_s_action == 1)[0]].shape[0])
		
		print ("Diff true", diff_true)
		print ("Diff pred", diff_pred)
		diff_class_true.append(diff_true)
		diff_class_pred.append(diff_pred)
		for j in range (support_dict['graph'][i].shape[1]):
			s_tokenized = tokenizer.convert_ids_to_tokens(support_dict['feature'][i].squeeze()[j].squeeze().tolist())
			print ("Support text", j, ":", s_tokenized)
			print ("S-Graph", j, ":", support_dict['graph'][i][0,j].nodes.data())
		print ("\n")
	return diff_class_true, diff_class_pred

#------- Analyze test results --------------#
def analyze_test_results(pred, query_class, tokenizer,	report_results):
	print ("------Incorrect -----")
	index_loc = np.argwhere(pred != query_class)
	copy_report = copy.deepcopy(report_results)
	copy_pred = copy.deepcopy(pred)
	copy_class = copy.deepcopy(query_class)
	diff_class_true, diff_class_pred = extract_subresults(copy_pred, copy_class, tokenizer, copy_report, index_loc)


	print ('-----Correct-------')
	index_loc = np.argwhere(pred == query_class)
	copy_report = copy.deepcopy(report_results)
	copy_pred = copy.deepcopy(pred)
	copy_class = copy.deepcopy(query_class)
	same_class_true, same_class_pred = extract_subresults(copy_pred, copy_class, tokenizer, copy_report, index_loc)

	return diff_class_true, diff_class_pred, same_class_true, same_class_pred

#new_original: [bsz,max_seg,D], new_mask: [bsz,max_seg]
def compute_pos_neg_similarity(new_original, new_pos, new_mask_original, new_mask_pos, transform_sim, conv1d, use_relu, naive_meanpool):
	#pos_matrix_mask = torch.matmul(new_mask_original.unsqueeze(-1), new_mask_pos.unsqueeze(-1).transpose(1,2)) # (bsz*bsz,original_seg,pos_seg)
	#pos_matrix_mask = new_mask_original.unsqueeze(-1) @ new_mask_pos.unsqueeze(-1).transpose(1,2)
	#pos_matrix_mask = torch.bmm(new_mask_original.unsqueeze(-1), new_mask_pos.unsqueeze(-1).transpose(1,2))
	num_act_sim = new_mask_original.sum(dim=-1, keepdim=True) * new_mask_pos.sum(dim=-1, keepdim=True)	
	original_input = new_original * new_mask_original.unsqueeze(-1)
	pos_input = new_pos * new_mask_pos.unsqueeze(-1)

	assert original_input.isnan().any() == False
	assert pos_input.isnan().any() == False
	prior_pos_sim = F.cosine_similarity(original_input.unsqueeze(2), pos_input.unsqueeze(1), dim=-1)
	assert prior_pos_sim.isnan().any() == False

	if (use_relu =='true'):
		inter_pos_sim = transform_sim(prior_pos_sim)
		print ("--Transform--")
	else:
		inter_pos_sim = prior_pos_sim

	assert inter_pos_sim.isnan().any() == False
	if (conv1d != None):
		print ("Convolution")
		conv_list = [F.relu(conv1d(inter_pos_sim)) for conv1d in conv1d]
		pool_list = [F.max_pool1d(x_conv, kernel_size = x_conv.shape[2]) for x_conv in conv_list] #each: [bsz, out_channels, 1]
		finalized_score = torch.cat([x_pool.squeeze(2) for x_pool in pool_list],dim=1) #[bsz,out_channels * len(num_filters)]
		pos_sim = torch.sum(finalized_score,dim=1, keepdim=True) #[bsz,1]
	else:
		if (naive_meanpool == 'true'):
			print ("Naive mean")
			pos_sim = inter_pos_sim.mean(dim=-1).mean(dim=-1,keepdim=True)
		elif (naive_meanpool == 'false'):
			pos_sim = inter_pos_sim.sum(dim=-1).sum(dim=-1,keepdim=True) / (num_act_sim +1e-20)
		else:
			print ("Max pooling")
			pos_sim = torch.amax(inter_pos_sim,-1).sum(dim=-1, keepdim=True) / (num_act_sim + 1e-20)
	
	assert pos_sim.isnan().any() == False	
	return pos_sim, inter_pos_sim, prior_pos_sim

def convert_tuple_list_to_input(config,i,counter, element_counter, input_emb, seg_mask, max_segment, max_seg_len):
	start_point = 1 # Do not take [CLS] into consideration
	seg_rep = []
	seg_full_rep = []
	seg_counter = 0
	cur_num = 0
	counter = 0
	for j in i:
		ori_start = j[0]
		ori_end = j[1]
		cur_input = input_emb[element_counter:element_counter+1, ori_start:ori_end+1] #1, len_of_specific_segment, D
		#Pad for full rep
		if (cur_input.shape[1] <max_seg_len):
			pad_zero = torch.zeros(1, max_seg_len - cur_input.shape[1], cur_input.shape[-1])
			pad_zero = move_to_cuda(config, pad_zero)[0]
			full_input = torch.cat([cur_input, pad_zero],1)
		else:
			full_input = cur_input
		seg_full_rep.append(full_input)
			
		cur_input = cur_input.mean(dim=1,keepdim=True) # mean over embedding of the tokens made up of that segment

		seg_rep.append(cur_input)
		cur_num += 1

	seg_full_rep = torch.cat(seg_full_rep, 0) # cur_#seg, max_seg_len,D

	seg_mask[element_counter,:cur_num] = 1.0
	element_counter += 1
	#Padding
	# Pad until it reaches the max_segment (pad by 0s)
	if (cur_num < max_segment):
		padded_outp = torch.zeros(1, max_segment-cur_num, input_emb[element_counter:element_counter+1,:,:].shape[-1])
		padded_outp = move_to_cuda(config, padded_outp)[0]
		seg_rep.append(padded_outp)

		# For full rep
		padded_len_outp = torch.zeros(max_segment-cur_num, max_seg_len, input_emb[element_counter:element_counter+1,:,:].shape[-1])
		padded_len_outp = move_to_cuda(config, padded_len_outp)[0]
		
		seg_full_rep = torch.cat([seg_full_rep, padded_len_outp],0) #max_seg_num, max_seg_len, D 
	seg_full_rep = seg_full_rep.unsqueeze(0) #1, max_seg_num, max_seg_len, D
	seg_rep = torch.cat(seg_rep,1)

	return seg_rep, seg_full_rep, seg_mask



def extract_rand_seg_text(input_text, segment_info, mapping):
	assert input_text.shape[0] == len(segment_info) # bsz the same
	all_seg_text = []
	all_seg_len=[]
	all_seg_len_std=[]
	for i in range (input_text.shape[0]):
		idx_info = segment_info[i]

		text = input_text[i]
		seg_text = []
		seg_len_for_sen=[]
		for seg in idx_info:
			start,end = seg[0], seg[1]
			
			if (start == end):
				output_text = text[start]
			else:
				output_text = ' '.join(text[start:end + 1]) # -1: account for [CLS] counting
			
			len_count = 0.0
			for t in output_text.split(' '):
				if not (t.startswith('##')):
					len_count += 1
			seg_len_for_sen.append(len_count)
			seg_text.append(output_text)
		
		seg_len_for_sen = np.array(seg_len_for_sen)
		avg_seg_len_for_sen, std_seg_len = np.mean(seg_len_for_sen), np.std(seg_len_for_sen)

		all_seg_len.append(avg_seg_len_for_sen)
		all_seg_len_std.append(std_seg_len)

		seg_text = np.array(seg_text, dtype='<U1024')
		all_seg_text.append(seg_text)
		
	return all_seg_text,all_seg_len,all_seg_len_std


def extract_segment_text(input_text, segment_info, mapping):
	assert input_text.shape[0] == len(segment_info) # bsz the same
	all_seg_text = []
	all_seg_len=[]
	all_seg_len_std=[]
	for i in range (input_text.shape[0]):
		idx_info = ast.literal_eval(segment_info[i][0])[0]

		text = input_text[i]
		seg_text = []
		seg_len_for_sen=[]
		for seg in idx_info:
			start,end = seg[0], seg[1]
			
			if (start == end):
				output_text = text[start]
			else:
				output_text = ' '.join(text[start:end + 1]) # -1: account for [CLS] counting
			
			len_count = 0.0
			for t in output_text.split(' '):
				if not (t.startswith('##')):
					len_count += 1
			seg_len_for_sen.append(len_count)
			#avg_seg_len_for_sen += len_count
			seg_text.append(output_text)
		
		seg_len_for_sen = np.array(seg_len_for_sen)
		avg_seg_len_for_sen, std_seg_len = np.mean(seg_len_for_sen), np.std(seg_len_for_sen)

		all_seg_len.append(avg_seg_len_for_sen)
		all_seg_len_std.append(std_seg_len)

		seg_text = np.array(seg_text, dtype='<U1024')
		all_seg_text.append(seg_text)
		
	return all_seg_text,all_seg_len,all_seg_len_std

def convert_chunk_to_break(text, chunk):
        all_texts = text.split(' ')
        #Example: book a restaurant ==> 0 1 since chunk: [a restaurant]

        all_chunks=[1]* (len(all_texts) - 1)
        for c in chunk:

                c=re.sub(r"^: ", '',c)	
                words = c.split(' ')
                #strip punc at the beginning
                first = words[0]
                last = words[-1]
                try:
                        first_idx = all_texts.index(first)
                except:
                        for t in all_texts:
                                if (t.startswith(first)):
                                        first_idx = all_texts.index(t)
                                        break
                                elif (t.endswith(first)):
                                        first_idx = all_texts.index(t)
                                        break
                try:
                        last_idx = all_texts.index(last)
                except:
                        for t in all_texts:
                                if (t.startswith(last)):
                                        last_idx = all_texts.index(t)
                                        break
                
                                elif (t.endswith(last)):
                                        last_idx = all_texts.index(t)
                                        break
                        last_idx = all_texts.index(last)
                all_chunks[first_idx: last_idx] = [1]*(last_idx - first_idx)
                if (first_idx > 0):
                        all_chunks[first_idx -1] = 0
                if (last_idx < len(all_texts) -2):
                        all_chunks[last_idx] = 0


        assert len(all_chunks) == len(all_texts) - 1
        return all_chunks

def extract_incorrect(focused_tie_label, full_tie_label, all_pred_label, all_seg_text):

	focused_tie_incorr=[]
	focused_break_incorr=[]

	full_tie_incorr=[]
	full_break_incorr=[]
		
	for idx in range (focused_tie_label.shape[0]):
		golden_label = np.array(focused_tie_label[idx])
		pred_label = np.array(all_pred_label[idx])

		focused_pred_label = pred_label
		seg_text = all_seg_text[idx]
		seg_elements = np.array(seg_text.split(' '),dtype='object')
		focused_incorrect_locs = np.argwhere(golden_label != focused_pred_label) # 
		
		focused_level_break_incorr = []
		focused_level_tie_incorr=[]
		for item in focused_incorrect_locs:

			gold = golden_label[item.item()]
			if (gold == 0):
				focused_level_break_incorr.append((seg_elements[item.item()], seg_elements[item.item()+1]))
			elif (gold == 1):
				focused_level_tie_incorr.append((seg_elements[item.item()], seg_elements[item.item()+1]))
			else: #-1 case => ignore
				pass
		focused_break_incorr.append(focused_level_break_incorr)
		focused_tie_incorr.append(focused_level_tie_incorr)
		full_golden_label = np.array(full_tie_label[idx])
		assert len(seg_elements) -1 == len(full_golden_label)
		full_incorrect_locs = np.argwhere(full_golden_label != pred_label) # 
			
		full_level_break_incorr = []
		full_level_tie_incorr=[]
		for item in full_incorrect_locs:
			gold = full_golden_label[item.item()]

			if (gold == 0):
				full_level_break_incorr.append((seg_elements[item.item()], seg_elements[item.item()+1]))
			else:

				full_level_tie_incorr.append((seg_elements[item.item()], seg_elements[item.item()+1]))

		full_break_incorr.append(full_level_break_incorr)
		full_tie_incorr.append(full_level_tie_incorr)

	return focused_break_incorr, focused_tie_incorr, full_break_incorr, full_tie_incorr



#--------#
def get_all_intent_slot_labels(all_paths):
		all_intents=[]
		all_slots=[]
		t_slot=[]
		for input_file in all_paths:
				dist_slot=set()
				with open(input_file, "r", encoding="utf-8") as f:
						reader = csv.reader(f, delimiter='\t')
						for i, line in enumerate(reader):
								intent = line[0].strip()
								if (intent not in all_intents):
										all_intents.append(intent)
								slots= line[2].strip().split(' ')
								text = line[1].strip()

								for s in slots:
										if (s not in all_slots):
												all_slots.append(s)
										if (s!= 'O'):
											name_slot = s.split('-')[1]
											dist_slot.add(name_slot)
				t_slot.append(dist_slot)
		all_intents, all_slots= sorted(all_intents), sorted(all_slots)
		return all_intents, all_slots

def move_to_cuda(config, *argv):
		cuda_available = torch.cuda.is_available()
		gpu_id = config['gpu_id']
		output=[]
		for arg in argv:
				if (cuda_available and gpu_id!=-1):
						tensor_arg = arg.to(gpu_id)
				output.append(tensor_arg)

		return tuple(output)


def convert_to_np(config, batch_index, *argv):
		output=[]
		for arg in argv:
				batch_sample = arg[batch_index]
				output.append(batch_sample)

		return tuple(output)
def convert_np_to_long_tensor(config, batch_index, *argv):
		cuda_available = torch.cuda.is_available()
		gpu_id = config['gpu_id']
		output=[]
		for arg in argv:
				batch_sample = arg[batch_index]
				tensor_arg = torch.Tensor(batch_sample).long()
				if (cuda_available and gpu_id!=-1):
						tensor_arg = tensor_arg.to(gpu_id)
				output.append(tensor_arg)

		return tuple(output)


def convert_full_np_to_long_tensor(config,  *argv):

		cuda_available = torch.cuda.is_available()
		gpu_id = config['gpu_id']
		output=[]
		for arg in argv:
				tensor_arg = torch.Tensor(arg).long()
				if (cuda_available and gpu_id!=-1):
						tensor_arg = tensor_arg.to(gpu_id)
				output.append(tensor_arg)

		return tuple(output)

def convert_np_to_long_tensor_dict(config, batch_index, all_slots, all_intents, *argv):
		cuda_available = torch.cuda.is_available()
		gpu_id = config['gpu_id']
		output=[]

		for arg in argv:
				batch_sample = arg[batch_index]
				tensor_arg = torch.Tensor(batch_sample).long()
				if (cuda_available and gpu_id!=-1):
						tensor_arg = tensor_arg.to(gpu_id)
				output.append(tensor_arg)
		output = (all_slots, all_intents) + output
		return load_helper.InputFeatures(tuples(output))

def conduct_emb(context_emb, val_emb, cont_val_emb, batch_slot_mask, batch_slot_labels_ids, batch_tokens, all_slots, config, slot_type_emb_dict):

	aggr_slot_emb = []
	for ex_idx in range (val_emb.shape[0]):
		cur_emb = val_emb[ex_idx]
		new_emb_np = copy.deepcopy(cur_emb.detach().cpu().numpy())
		new_emb = torch.Tensor(new_emb_np)
		new_emb = move_to_cuda(config, new_emb)[0]
		
		slot_mask, slot_labels_id = batch_slot_mask[ex_idx], batch_slot_labels_ids[ex_idx]
		cur_token = batch_tokens[ex_idx]
		#Iterate through each slot

		unique_all_slots, unique_all_counts = np.unique(slot_mask, return_counts=True)
		unique_slots, unique_counts = unique_all_slots[2:], unique_all_counts[2:]

		for idx in range (len(unique_slots)): #[1,2,3...]
				cur_slot, cur_slot_count = unique_slots[idx].item(), unique_counts[idx].item() #artificial count

				slot_idx = np.argwhere(slot_mask == cur_slot).reshape(-1,)
				head_idx = slot_idx[0].item()

				full_name = all_slots[slot_labels_id[head_idx]]
				b_tok, name = full_name.split('-')
				if (slot_labels_id[head_idx] in slot_type_emb_dict['context']):
					cur_slot_emb = context_emb[ex_idx,head_idx,:]

				elif (slot_labels_id[head_idx] in slot_type_emb_dict['val']):
					cur_slot_emb = val_emb[ex_idx, head_idx,:]
				else:
					cur_slot_emb = cont_val_emb[ex_idx, head_idx,:]

				new_rep = cur_slot_emb
				new_emb[head_idx] = new_rep

		aggr_slot_emb.append(new_emb)
	aggr_slot_emb = torch.stack(aggr_slot_emb, dim=0) # [bsz, max_len,D]
	return aggr_slot_emb

def compute_slot_emb(embed_x, batch_slot_mask, batch_slot_labels_ids, batch_tokens, all_slots, config, slot_for_plot=None, slot_for_val=None, slot_type_emb_dict=None):

	aggr_slot_emb = []
	for ex_idx in range (embed_x.shape[0]):
		cur_emb = embed_x[ex_idx]

		new_emb_np = copy.deepcopy(cur_emb.detach().cpu().numpy())
		new_emb = torch.Tensor(new_emb_np)
		new_emb = move_to_cuda(config, new_emb)[0]
		
		slot_mask, slot_labels_id = batch_slot_mask[ex_idx], batch_slot_labels_ids[ex_idx]
		cur_token = batch_tokens[ex_idx]
		#Iterate through each slot

		unique_all_slots, unique_all_counts = np.unique(slot_mask, return_counts=True)
		unique_slots, unique_counts = unique_all_slots[2:], unique_all_counts[2:]
		for idx in range (len(unique_slots)): #[1,2,3...]
				cur_slot, cur_slot_count = unique_slots[idx].item(), unique_counts[idx].item() #artificial count

				slot_idx = np.argwhere(slot_mask == cur_slot).reshape(-1,)
				head_idx = slot_idx[0].item()
				if (config['context'] == 'context-val(context)'): # 
						#Average all of the remaining words
						cur_slot_emb = cur_emb[(slot_mask != -1) & (slot_mask != cur_slot)] # tensor
						assert cur_token.shape[0] -2 - len(slot_idx) == cur_slot_emb.shape[0]
						cur_slot_emb = cur_slot_emb.mean(dim=0) # 1,D
				elif (config['context'] == 'context-val(adj-context)'):
						#Adjacent Calculation
						start, end = 0, 0
						tail_idx = slot_idx[-1].item()
						if (slot_idx[0].item() == 1): # first token
								start = -1
						if (slot_idx[-1].item()  == cur_token.shape[0] -2): #last token before [SEP]
								end = -1

						if (start !=-1 and end != -1): # normal case
								cur_slot_emb = torch.cat([cur_emb[head_idx -1:head_idx],cur_emb[tail_idx +1: tail_idx+2]], 0)
								cur_slot_emb = cur_slot_emb.mean(dim=0)
						elif (start == -1):
								cur_slot_emb = cur_emb[tail_idx + 1]
						elif (end == -1):

								cur_slot_emb = cur_emb[head_idx-1]

				else:
						assert cur_slot_count == len(slot_idx)
						cur_slot_emb = cur_emb[slot_idx,:][0] #only use the first token #[D]

				new_rep = cur_slot_emb
				new_emb[head_idx] = new_rep
				full_name = all_slots[slot_labels_id[head_idx]]

				b_tok, name = full_name.split('-')
				assert b_tok == 'B'
				if (slot_for_plot != None):
					if (name not in slot_for_plot):
							slot_for_plot[name] = [cur_slot_emb.detach().cpu().numpy()]
					else:
							slot_for_plot[name].append(cur_slot_emb.detach().cpu().numpy())
					val = list(cur_token[slot_idx])
					val = ' '.join(val)

					if (name not in slot_for_val):
							slot_for_val[name] = [val]
					else:
							slot_for_val[name].append(val)
				else:
					slot_for_plot=None
					slot_for_val=None
		aggr_slot_emb.append(new_emb)
	aggr_slot_emb = torch.stack(aggr_slot_emb, dim=0) # [bsz, max_len,D]
	if (config['context'] != 'context-val(context)' and config['context'] != 'context-val(adj-context)'):
			assert torch.allclose(aggr_slot_emb, embed_x)
	return aggr_slot_emb, slot_for_plot, slot_for_val

