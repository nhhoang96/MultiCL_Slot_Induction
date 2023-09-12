import numpy as np
import torch
import copy
import torch.nn.functional as F
import torch.nn as nn
import math
import unicodedata
from transformers import *

from utils import utils
import time

model_dir = '../bert_model/'
tokenizer = BertTokenizer.from_pretrained(model_dir)
mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]


special_tokens = ['-', '&', ':', "'", '’']
special_id = []
for t in special_tokens:
	t_id = tokenizer.convert_tokens_to_ids([t])[0]
	special_id.append(t_id)


def original_cleanup(init_matrix, mapping, indexed_tokens, tokenized_text, subwords):
	simplified_toks = [indexed_tokens[i] for i in range (len(indexed_tokens)) if i not in subwords]
	simplified_toks = simplified_toks[1:-1]

	simplified_text = [tokenized_text[i] for i in range (len(tokenized_text)) if i not in subwords]
	simplified_text = simplified_text[1:-1]
	merge_column_matrix = []
	for i, line in enumerate(init_matrix):
		new_row = []
		buf = []
		for j in range(0, len(line) - 1):
			buf.append(line[j])
			if (mapping[j] != mapping[j+1]):
				new_row.append(buf[0])
				buf = []
		merge_column_matrix.append(new_row)

	# merge subwords in multi rows
	# transpose the matrix so we can work with row instead of multiple rows
	merge_column_matrix = np.array(merge_column_matrix).transpose()
	merge_column_matrix = merge_column_matrix.tolist()
	final_matrix = []
	for i, line in enumerate(merge_column_matrix):
		new_row = []
		buf = []
		for j in range(0, len(line) - 1):
			buf.append(line[j])
			if mapping[j] != mapping[j + 1]:
				new_row.append(buf[0])
				buf = []
		final_matrix.append(new_row)

	# transpose to the original matrix
	final_matrix = np.array(final_matrix).transpose()
	assert final_matrix.shape[0] == final_matrix.shape[1]
	final_matrix = final_matrix[1:, 1:]
	return final_matrix,simplified_toks, simplified_text


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
		return tmp, current_mask


#----- Perturbed Masking utils ------
#---Preprocessing Utils (Subotken re-matching) -----#

#Original Mapping from Perturbed Paper ([-1, 0, 1 , 1, 2,-1])-----#
def _run_strip_accents(text):
	"""Strips accents from a piece of text."""
	text = unicodedata.normalize("NFD", text)
	output = []
	for char in text:
			cat = unicodedata.category(char)
			if cat == "Mn":
					continue
			output.append(char)
	return "".join(output)

def match_tokenized_to_untokenized(subwords, sentence):
	token_subwords = np.zeros(len(sentence))
	sentence = [_run_strip_accents(x) for x in sentence]
	token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
	for i, subword in enumerate(subwords):
		if subword in ["[CLS]", "[SEP]"]: continue

		while current_token_normalized is None:
			current_token_normalized = sentence[current_token].lower()

		if subword.startswith("[UNK]"):
			if (subword[6:] == ''):
					pass
			else:
					unk_length = int(subword[6:])
					subwords[i] = subword[:5]
					subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
		else:
				subwords_str += subword[2:] if subword.startswith("##") else subword
		if not current_token_normalized.startswith(subwords_str):
			#print ("Subword str", subwords_str)
			#print ("Normalized token", current_token_normalized)
			return False

		token_ids[i] = current_token
		token_subwords[current_token] += 1

		if current_token_normalized == subwords_str:
			subwords_str = ""
			current_token += 1
			current_token_normalized = None

	assert current_token_normalized is None
	while current_token < len(sentence):
		assert not sentence[current_token]
		current_token += 1
	assert current_token == len(sentence)

	return token_ids



# Subword mapping ([[], [1,2], [3], []]) -----#
def _is_special(ch):
	return bool(ch) and (ch[0] =='[') and (ch[-1] ==']')

def _is_control(ch):
	return unicodedata.category(ch) in ('Cc', 'Cf')

def stem(token):
	if (token[:2] == '##'):
		return token[2:]
	else:
		return token
def rematch(text, tokens):
	split_text = text.split(' ')
	normalized_text, char_mapping = '', []
	for i, ch in enumerate(text):
		ch = unicodedata.normalize('NFD', ch)
		ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
		ch = ''.join([
				c for c in ch
				if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
		])
		normalized_text += ch
		char_mapping.extend([i]* len(ch))
	char_mapping = [i for i in range(len(tokens))]
	text, token_mapping, offset = normalized_text, [], 0
	counter =0
	prev_token = None

	dangerous_tokens=['-', '&',':',"'", '’']
	for token in tokens:
		if _is_special(token):
			token_mapping.append([])
		else:
			start = tokens[offset:].index(token) + offset
			end = start + 1	
			if (token.startswith('##')):
				token_mapping[counter].append(char_mapping[start:end][0])
			else:

				token_mapping.append(char_mapping[start:end])
				counter += 1
			offset = end
			prev_token = token
	return token_mapping



# Old Mapping ----#


def cleanup(init_matrix, tokenized_text, indexed_tokens, subwords):
	if (len(subwords) > 0):
		init_matrix = np.delete(np.delete(init_matrix,subwords,0),subwords,1) # Remove subwords
	init_matrix = np.delete(np.delete(init_matrix, [0,-1],1),[0,-1], 0) # Remove CLS and SEP token
	simplified_text = [tokenized_text[i] for i in range (len(tokenized_text)) if i not in subwords]
	simplified_text = simplified_text[1:-1]
	
	simplified_toks = [indexed_tokens[i] for i in range (len(indexed_tokens)) if i not in subwords]
	simplified_toks = simplified_toks[1:-1]

	return init_matrix, simplified_text, simplified_toks

def decode_subword_matrix(init_matrix, tokenized_text, indexed_tokens, subwords, mapping):
	merge_column_matrix = []
	for i, line in enumerate(init_matrix):
		new_row = []
		buf = []
		for j in range(0, len(line) - 1):
			buf.append(line[j])
			if mapping[j] != mapping[j + 1]:
				new_row.append(buf[0])
				buf = []
			merge_column_matrix.append(new_row)

	# merge subwords in multi rows
	# transpose the matrix so we can work with row instead of multiple rows
	merge_column_matrix = np.array(merge_column_matrix).transpose()
	merge_column_matrix = merge_column_matrix.tolist()
	final_matrix = []
	for i_merge, line_merge in enumerate(merge_column_matrix):		
		new_row = []
		buf = []
		for j_merge in range(0, len(line_merge) - 1):
			buf.append(line_merge[j_merge])
			if mapping[j_merge] != mapping[j_merge + 1]:
				new_row.append(buf[0])
				buf = []
		final_matrix.append(new_row)

	# transpose to the original matrix
	final_matrix = np.array(final_matrix).transpose()

	# filter some empty matrix (only one word)
	assert final_matrix.shape[0] == final_matrix.shape[1]	
	final_matrix = softmax(final_matrix)

	np.fill_diagonal(final_matrix, 0.)

	final_matrix = 1. - final_matrix
	np.fill_diagonal(final_matrix, 0.)

	simplified_text = [tokenized_text[i] for i in range (len(tokenized_text)) if i not in subwords]
	simplified_text = simplified_text[1:-1]
	
	simplified_toks = [indexed_tokens[i] for i in range (len(indexed_tokens)) if i not in subwords]
	simplified_toks = simplified_toks[1:-1]
	return final_matrix,simplified_text, simplified_toks

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def clean_cls(text):
		return_text = text.replace('[CLS] ', '')
		return_text = return_text.replace(' [SEP]', '')
		return return_text


def prepare_tensor(config, mapping, indexed_tokens,tokenized_text, current_index):
        id_for_all_i_tokens = get_all_subword_id(mapping, current_index)
        tmp_indexed_tokens = list(indexed_tokens)
        for tmp_id in id_for_all_i_tokens:
                        if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                                        tmp_indexed_tokens[tmp_id] = mask_id
        one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
        for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                        if mapping[tmp_id] != -1:
                                one_batch[j][tmp_id] = mask_id
        # 2. Convert one batch to PyTorch tensors
        tokens_tensor = torch.tensor(one_batch)
        segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])

        tokens_tensor, segments_tensor = utils.move_to_cuda(config, tokens_tensor, segments_tensor)
        return tokens_tensor, segments_tensor


# For a segment made up from multiple tokens, index is segment index (i.e. (1,1) or (2,5))
def prepare_segment_tensor(config, mapping, indexed_tokens,tokenized_text, current_index, segment_list):
	start, end = segment_list[current_index][0], segment_list[current_index][1]
	id_for_all_i_tokens = []
	for idx in range (start, end+1):
		current_tokens = get_all_subword_id(mapping, idx)
		id_for_all_i_tokens.extend(current_tokens)
	tmp_indexed_tokens = list(indexed_tokens)
	for tmp_id in id_for_all_i_tokens:
		if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
			tmp_indexed_tokens[tmp_id] = mask_id
	one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
	for j in range(0, len(segment_list)):
	#for j in range(0, len(tokenized_text)):
		id_for_all_j_tokens=[]
		start, end = segment_list[j][0], segment_list[j][1]
		#print ("St", start, end)
		for idx in range (start, end+1):
			cur_toks = get_all_subword_id(mapping, idx)
			id_for_all_j_tokens.extend(cur_toks)
		for idx in range (start, end+1):
			for tmp_id in id_for_all_j_tokens:
				if mapping[tmp_id] != -1:
					one_batch[idx][tmp_id] = mask_id
	# 2. Convert one batch to PyTorch tensors
	tokens_tensor = torch.tensor(one_batch)
	segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
	tokens_tensor, segments_tensor = utils.move_to_cuda(config, tokens_tensor, segments_tensor)
	return tokens_tensor, segments_tensor

def getdepth(tree):		 
	if not(isinstance(tree, list)):
		return 0
	leftdepth = getdepth(tree[0])
	rightdepth = getdepth(tree[1])
	if (leftdepth > rightdepth):
		return leftdepth + 1
	else:
		return rightdepth + 1


def find_best_tie(scores):
	best_score = -np.inf
	best_cut = -1
	tie_pos = []
	for k in range(1, len(scores)):
		cut_score = calculate_tie_score(k, scores)
		if cut_score > best_score:
			best_cut = k
			best_score = cut_score
	
	tie_pos.append(best_cut)
	return best_cut, best_score, tie_pos

def find_tie_scores(scores):
	best_score = -np.inf
	best_cut = -1
	tie_pos = []
	all_tie_scores=[]
	for k in range(1, len(scores)):
		tie_score = calculate_tie_score(k, scores)
		if tie_score > best_score:
			best_cut = k
			best_score = tie_score
		all_tie_scores.append(tie_score)
	tie_pos.append(best_cut)
	return all_tie_scores


def calculate_tie_score(k,scores):
	#Sum I(2,1) and I(1,2) and maximize this summation
	surr = np.sum(scores[k-1:k+1, k-1:k+1])
	tie_score = surr/ (2 + 1e-20)
	return tie_score

#----- Obtain MART SCORE ----#
def mart_tie(scores, sen):
	assert len(scores) == len(sen)
	score_split=[]	
	if len(scores) == 1:
		parse_tree = sen[0]
		score_split = []

	else:
		idx_max, score_max = find_best_cut(scores)
		parse_tree = []
		score_split.append(score_max)
		if len(sen[:idx_max]) > 0:
			tree0,score0 = mart(scores[:idx_max, :idx_max], sen[:idx_max])
			parse_tree.append(tree0)
			score_split.append(score0)
		tree1 = sen[idx_max]
		score2 = score_max
		if len(sen[idx_max + 1:]) > 0:

			tree2,score2 = mart(scores[idx_max + 1:, idx_max + 1:], sen[idx_max + 1:])
			tree1 = [tree1, tree2]

		if parse_tree == []:
			parse_tree = tree1
			score_split = score2
		else:
			parse_tree.append(tree1)
			score_split.append(score2)

	return parse_tree,score_split

def find_cut_scores(scores, new_score,type_score):
	best_score = np.inf
	best_cut = -1
	all_cut_scores = []
	for k in range(1, len(scores)):
		cut_score = calculate_test_cut_score(k, scores, new_score, type_score)
		if cut_score < best_score:
			best_cut = k
			best_score = cut_score
		all_cut_scores.append(cut_score)
	return all_cut_scores

def find_best_cut2(scores, new_score, type_score):
	best_score = np.inf
	best_cut = -1
	for k in range(1, len(scores)):
		cut_score = calculate_test_cut_score(k, scores, new_score,type_score)
		if cut_score < best_score:
			best_cut = k
			best_score = cut_score
	return best_cut, best_score

def find_best_cut(scores, new_score):
	best_score = np.inf
	best_cut = -1
	for k in range(1, len(scores)):
		cut_score = calculate_cut_score(k, scores, new_score)
		if cut_score < best_score:
			best_cut = k
			best_score = cut_score
	return best_cut, best_score


def calculate_test_cut_score(k,scores, new_score, type_score):

	if (type_score == 'original'):
		sq1 = 2 * k
		sq2 = 2 * (len(scores) - k)
		rec = (len(scores) - k) * k 

		left = np.sum(scores[:k, :k]) / (sq1 + 1e-20)
		right = np.sum(scores[k + 1:, k + 1:]) / (sq2 + 1e-20)

		between = np.sum(scores[:k, k + 1:]) + np.sum(scores[k + 1:, :k])
		between /= (rec+ 1e-20)

		cut_score = left + right - between

	elif (type_score == 'full'):
		sq1 = k*(k+1)
		sq2= (len(scores) -1 -k) * (len(scores) -k -2)
		rec = 2*(len(scores) -1 -k) * (k+1)
		
		w_sq1 = 2 * k
		w_sq2 = 2 * (len(scores) - k)
		w_rec = (len(scores) - k) * k 


		new_sq1 = k*(k-1)
		new_sq2 = (len(scores) -k -1) * (len(scores) - k -2)

		new_rec = 2*(len(scores) - 1 - k) * k 
		unaccounted = 2*(len(scores) -1)

		assert sq1 + sq2 + rec + len(scores) == len(scores) **2
		assert new_sq1+new_sq2+new_rec+unaccounted+len(scores) == len(scores) **2
		if (new_score == 'true'): #new == not include green
			new_rec += unaccounted
		
		left = np.sum(scores[:k, :k]) / (new_sq1 + 1e-20)
		right = np.sum(scores[k + 1:, k + 1:]) / (new_sq2 + 1e-20)


		assert 0<= left <= 1
		assert 0<= right <= 1


		if (new_score == 'true'):
			between = np.sum(scores[:k+1, k + 1:]) + np.sum(scores[k + 1:, :k+1])
		else:
			between = np.sum(scores[:k, k + 1:]) + np.sum(scores[k + 1:, :k])

		between /= (new_rec + 1e-20)
		assert 0 <= between <=1

		cut_score = left + right - between
	
	elif (type_score == 'change_bet'):
		#---Original denom ---#
		sq1 = 2 * k
		sq2 = 2 * (len(scores) - k)
		rec = (len(scores) - k) * k 

		left = np.sum(scores[:k, :k]) / (sq1 + 1e-20)
		right = np.sum(scores[k + 1:, k + 1:]) / (sq2 + 1e-20)

		right_word_impact_count= 2*(len(scores) - 1-k)#for single row + single column
		right_word_impact = np.sum(scores[k:k+1,k+1:]) + np.sum(scores[k+1:,k:k+1]) # impact on the right phrases => Distance needs to be small
		
		left_word_impact_count= 2*k #for single row + single column
		left_word_impact = np.sum(scores[k:k+1,:k]) + np.sum(scores[:k,k:k+1]) # impact on the left phrases => Distance needs to be high for separation
		
		between = np.sum(scores[:k+1, k + 1:]) + np.sum(scores[k + 1:, :k+1])
		right_word_impact = right_word_impact / (right_word_impact_count + 1e-20)
		left_word_impact = left_word_impact / (left_word_impact_count + 1e-20)

		cut_score = left + right + right_word_impact -left_word_impact - between

	elif (type_score == 'no_bet'):
		sq1 = 2 * k
		sq2 = 2 * (len(scores) - k)
		rec = (len(scores) - k) * k 

		left = np.sum(scores[:k, :k]) / (sq1 + 1e-20)
		right = np.sum(scores[k + 1:, k + 1:]) / (sq2 + 1e-20)

		cut_score = left + right	
	return cut_score



def calculate_cut_score(k,scores, new_score):
	sq1 = 2 * k
	sq2 = 2 * (len(scores) - k)
	rec = (len(scores) - k) * k 

	
	left = np.sum(scores[:k, :k]) / (sq1 + 1e-20)
	right = np.sum(scores[k + 1:, k + 1:]) / (sq2 + 1e-20)
	between = np.sum(scores[:k, k + 1:]) + np.sum(scores[k + 1:, :k])
	between /= (rec + 1e-20)

	total = sq1+ sq2 + rec
	ratio_left = sq1 / (1e-20 + total)
	ratio_right = sq2/ (1e-20 + total)
	ratio_bet = rec / (1e-20 + total)
	if (new_score == 'false'):

		cut_score = left + right - between
	else:
		cut_score = ratio_left * left + ratio_right * right - ratio_bet * between
	
	return cut_score


#----- Obtain MART SCORE ----#
def mart2(scores, sen,new_score, score_type):
	assert len(scores) == len(sen)
	score_split=[]	
	if len(scores) == 1:
		parse_tree = sen[0]
		score_split = []

	else:
		idx_max, score_max = find_best_cut2(scores, new_score, score_type)
		parse_tree = []
		score_split.append(score_max)
		if len(sen[:idx_max]) > 0:
			tree0,score0 = mart(scores[:idx_max, :idx_max], sen[:idx_max], new_score)
			parse_tree.append(tree0)
			score_split.append(score0)
		tree1 = sen[idx_max]
		score2 = score_max
		if len(sen[idx_max + 1:]) > 0:

			tree2,score2 = mart(scores[idx_max + 1:, idx_max + 1:], sen[idx_max + 1:], new_score)
			tree1 = [tree1, tree2]

		if parse_tree == []:
			parse_tree = tree1
			score_split = score2
		else:
			parse_tree.append(tree1)
			score_split.append(score2)

	return parse_tree,score_split



#----- Obtain MART SCORE ----#
def mart(scores, sen,new_score):
	assert len(scores) == len(sen)
	score_split=[]	
	if len(scores) == 1:
		parse_tree = sen[0]
		score_split = []

	else:
		idx_max, score_max = find_best_cut(scores, new_score)
		parse_tree = []
		score_split.append(score_max)
		if len(sen[:idx_max]) > 0:
			tree0,score0 = mart(scores[:idx_max, :idx_max], sen[:idx_max], new_score)
			parse_tree.append(tree0)
			score_split.append(score0)
		tree1 = sen[idx_max]
		score2 = score_max
		if len(sen[idx_max + 1:]) > 0:

			tree2,score2 = mart(scores[idx_max + 1:, idx_max + 1:], sen[idx_max + 1:], new_score)
			tree1 = [tree1, tree2]

		if parse_tree == []:
			parse_tree = tree1
			score_split = score2
		else:
			parse_tree.append(tree1)
			score_split.append(score2)

	return parse_tree,score_split

def removeNestings(l, output):
	for i in l:
		if type(i) == list:
			 removeNestings(i,output)
		else:
			 output.append(i)
	return output

def removeNestings_id(l, output):
	for i in l:
		if type(i) == list:
			removeNestings(i,output)
		else:
			output.append(i)
	return output


def get_key(val,id_dict):
	r = []
	r = removeNestings(val,r) # flatten the list
	return_idx=[]
	for idx in range (len(r)):
		if (isinstance(id_dict[r[idx]],list)): # multiple occurences
			return_idx.append(id_dict[r[idx]][0])
		else:
			return_idx.append(id_dict[r[idx]])
	return return_idx


def addNestings(l, output):
	for i in l:
		if type(i) == list:
			 addNestings(i,output)
		else:
			 output.append([i])
	return output

def get_all_subword_id(mapping, idx):
	current_id = mapping[idx]
	id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
	return id_for_all_subwords


#---eld Mapping -----
def _run_strip_accents(text):
	"""Strips accents from a piece of text."""
	text = unicodedata.normalize("NFD", text)
	output = []
	for char in text:
			cat = unicodedata.category(char)
			if cat == "Mn":
					continue
			output.append(char)
	return "".join(output)

def match_tokenized_to_untokenized(subwords, sentence):
	token_subwords = np.zeros(len(sentence))
	sentence = [_run_strip_accents(x) for x in sentence]
	token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
	for i, subword in enumerate(subwords):
		if subword in ["[CLS]", "[SEP]"]: continue

		while current_token_normalized is None:
			current_token_normalized = sentence[current_token].lower()

		if subword.startswith("[UNK]"):
			if (subword[6:] == ''):
				pass
			else:
				unk_length = int(subword[6:])
				subwords[i] = subword[:5]
				subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
		else:
			subwords_str += subword[2:] if subword.startswith("##") else subword

		if not current_token_normalized.startswith(subwords_str):
			#print ("Subword str", subwords_str)
			#print ("Normalized token", current_token_normalized)
			return False

		token_ids[i] = current_token
		token_subwords[current_token] += 1
		if current_token_normalized == subwords_str:
			subwords_str = ""
			current_token += 1
			current_token_normalized = None

	assert current_token_normalized is None
	while current_token < len(sentence):
		assert not sentence[current_token]
		current_token += 1
	assert current_token == len(sentence)

	return token_ids


def getrep_id(k,tree, depth):
	if (k == 0):
		out = []
		out= removeNestings(tree, out)
		output=out
	elif (k == depth):
		out = []
		out= removeNestings(tree, out)
		output = [[o] for o in out]
	else: #Level >=2
		if (not isinstance(tree[0],list)):
			output= [[tree[0]]]
			nextLevel = tree[1]
		else:
			output=[]
			nextLevel= [tree[0], tree[1]]
		level = 0
		while (level < k):
			_nextLevel = []
			_currentLevel = []
			for subtree in nextLevel:
				#Preserve the remaining at the current depth (last level) 
				if (level == k - 1): #except for depth = 2
					if (isinstance(subtree, list)):
						output.append(subtree)
					else:
						output.append([subtree])
				else:
					if (isinstance(subtree, list)):
						if ((isinstance(subtree[0], list)) and (isinstance(subtree[1],list))):
							_nextLevel.append(subtree[0])	
							_nextLevel.append(subtree[1])	
						elif (not (isinstance(subtree[0], list)) and (isinstance(subtree[1],list))):
							output.append([subtree[0]])
							_nextLevel.append(subtree[1])
							
						elif ((isinstance(subtree[0], list)) and (not isinstance(subtree[1],list))):

							output.append([subtree[1]])
							_nextLevel.append(subtree[0])
						else:
							output.append([subtree[0]])
							output.append([subtree[1]])
						
					else:
						_currentLevel.append(subtree)
						output.append([subtree])
			nextLevel = _nextLevel
			currentLevel = _currentLevel
			level += 1
		if (nextLevel != []): 
			if (isinstance(nextLevel, list)):
				append_val = []
				append_val = removeNestings(nextLevel, append_val)
				output.append(append_val)
			else:
				output.append([nextLevel])
	return output


def reposition(split_loc, new_report_list, prev_report_list, single_item_misplaced, item_keys):
	if (len(split_loc) > 0):
		duplicate=[]
		copy_prev_list = copy.deepcopy(prev_report_list)
		for n_index in range (len(split_loc)):	
			cur_loc = split_loc[n_index]
			cur_comb = new_report_list[cur_loc] + new_report_list[cur_loc +1]
			if (cur_comb in prev_report_list):	
				supposed, copy_prev_list, duplicate = obtain_supposed_loc(cur_comb, prev_report_list, copy_prev_list, duplicate, split_loc, new_report_list, cur_loc)
				item_keys[supposed] = new_report_list[cur_loc]
				item_keys[supposed+1] = new_report_list[cur_loc + 1]
	return new_report_list, item_keys

def adjust_end_point_single(subword_mapping, start_point):
	val_at_start_point = subword_mapping[start_point]	
	val_in_mapping, val_in_count = np.unique(np.array(subword_mapping), return_counts=True)
	subword_idx  = val_in_count[list(val_in_mapping).index(val_at_start_point)].item()
	if (subword_idx > 1):
		end_point = start_point + (subword_idx-1)
	else:
		end_point = start_point

	return end_point

def handle_start_point(id_dict, occ_freq,prev_freq, start_token, last_end_point):
	cur_start_idx = 0
	if (len(list(prev_freq.keys()))>=1):
		if (start_token in prev_freq):
			cur_start_idx = occ_freq[start_token] + prev_freq[start_token]
			start_point = id_dict[start_token][occ_freq[start_token] + prev_freq[start_token]]
		else:	
			start_point = id_dict[start_token][occ_freq[start_token]]	
			cur_start_idx = occ_freq[start_token] 

	else:
		start_point = id_dict[start_token][occ_freq[start_token]]	
		cur_start_idx = occ_freq[start_token]
	if (last_end_point > start_point and len(id_dict[start_token]) > 1):
		start_point = id_dict[start_token][cur_start_idx + 1]
	return start_point

def obtain_supposed_loc(cur_comb, prev_report_list, copy_prev, duplicate, split_loc, new_report_list, cur_loc):
	if (cur_comb not in duplicate):	
		prev_pos = prev_report_list.index(cur_comb)
		copy_prev[prev_pos] = -1

	else:
		#Duplicate means it has been a single segment before, find the next place this single segment occurs
		prev_pos = prev_report_list.index(cur_comb)	

		if (cur_comb in copy_prev):
			prev_pos = copy_prev.index(cur_comb)
			copy_prev[prev_pos] = -1

	duplicate.append(cur_comb)

	pad = 0
	for j in range (prev_pos):
		prev_comb_item = prev_report_list[j]
		for k in split_loc:
			item = new_report_list[k] + new_report_list[k+1] 
			if (item == prev_comb_item):
				pad += 1
				break

	supposed = prev_pos + pad
	return supposed, copy_prev,duplicate

def create_level_dict(new_tree, level, id_dict, subwords, subword_mapping, tokenized_text, indexed_tokens, scores,num_occ_dict, new_score,desired_lev=-1):
	if (len(new_tree) == 1):
		new_tree_depth = 0
	else:	
		new_tree_depth = getdepth(new_tree)

	if (desired_lev != -1):
		if (desired_lev >= new_tree_depth):
			desired_lev = new_tree_depth
		
	return_level = None	
	return_dict ={}
	prev_report_list = None
	# For each level
	for i in range (new_tree_depth + 1):
		extracted_level = i
		current_level = []

		cut_scores = []
		return_list = getrep_id(extracted_level,new_tree, new_tree_depth)
		if (extracted_level == 0):
			report_list = []
			report_list = removeNestings(return_list, report_list)
			start_point = id_dict[report_list[0]][0]
			end_point = id_dict[report_list[-1]][-1]

			val_at_end_point = subword_mapping[end_point]	
			val_in_mapping, val_in_count = np.unique(np.array(subword_mapping), return_counts=True)
			
			subword_idx  = val_in_count[list(val_in_mapping).index(val_at_end_point)].item()
			if (subword_idx > 1):
				new_end_point = end_point + (subword_idx-1)
			else:
				new_end_point = end_point
			current_level = [(start_point, new_end_point)]
		else:
			if (extracted_level != new_tree_depth):
				report_list = []
				counter = 0
				for e in return_list:
					if (e ==101 or e==102):
						report_list.append(e)
					else:
						r = []
						r = removeNestings_id(e, r)
						report_list.append(r)
					counter += 1
			else:
				report_list = return_list
			
			if (extracted_level == new_tree_depth):
				pass
			else:
				copy_id_dict = copy.deepcopy(id_dict)
				new_report_list= [0]*len(report_list)
				need_sort=[]
				locs = []
				prev_mismatch=[]
				expansion_locs =[]
				split = False
				if (prev_report_list !=None):
					for index in range (len(report_list)):

						if (index < len(prev_report_list) and split == False):	
							if (report_list[index] == prev_report_list[index]):
								need_sort.append(report_list[index])

								locs.append(index)		
								new_report_list[index] = report_list[index]
							else:
								locs.append(index)		
								need_sort.append(report_list[index])
								prev_mismatch.append(prev_report_list[index])
								expansion_locs.append(index)
								split=True
						else: # Final index
							if (index < len(prev_report_list)):
								prev_mismatch.append(prev_report_list[index])
							locs.append(index)
							need_sort.append(report_list[index])
				
					need_sort = sorted(need_sort, key=lambda x: get_key(x,copy_id_dict)) # rearranging segments by the first token location in a sentence
					split_loc = []
					duplicate_log = []
					s_index = 0
					single_item_misplaced =set()
					copy_prev_list = copy.deepcopy(prev_report_list)

					item_keys={}
					while (s_index <= len(need_sort)-1):
						if (need_sort[s_index]  in copy_prev_list): # single item		
							single_item_misplaced.add(locs[s_index])
							copy_prev_list.remove(need_sort[s_index])

						for j_index in range (s_index,len(need_sort)):
							combined_item = need_sort[s_index] + need_sort[j_index]

							reverse_combined_item = need_sort[j_index] + need_sort[s_index]
							if (combined_item in prev_report_list):
								if ((combined_item not in duplicate_log) or (combined_item in duplicate_log and prev_report_list.count(combined_item) > 1)):
									tmp1 = need_sort[s_index+1]
									need_sort[s_index+1] = need_sort[j_index]
									need_sort[j_index] = tmp1
									split_loc.append(locs[s_index])
									duplicate_log.append(combined_item)
									if (j_index not in single_item_misplaced):
										single_item_misplaced.add(j_index)

									s_index+= 1
									break
							elif (reverse_combined_item in prev_report_list):

								if ((reverse_combined_item not in duplicate_log) or (reverse_combined_item in duplicate_log and prev_report_list.count(reverse_combined_item) > 1)):
									tmp = need_sort[j_index]
									need_sort[j_index] = need_sort[s_index]
									need_sort[s_index] = tmp
									
									tmp1 = need_sort[s_index+1]
									need_sort[s_index+1] = need_sort[j_index]
									need_sort[j_index] = tmp1

									duplicate_log.append(reverse_combined_item)
									split_loc.append(locs[s_index])


									if (j_index not in single_item_misplaced):
										single_item_misplaced.add(j_index)
									s_index+= 1
									break
						s_index+=1		
					single_item_misplaced = list(single_item_misplaced)

					for l_index in range (len(locs)):
						new_report_list[locs[l_index]] = need_sort[l_index]
	
					# Find supposed position of single element
					split_loc = sorted(split_loc)
					duplicate=[]
					copy_prev = copy.deepcopy(prev_report_list)
					for idx in range (len(single_item_misplaced)): # Each element is a single element (not as a pair with anyting before or after)
						cur_loc = single_item_misplaced[idx]

						cur_comb = new_report_list[cur_loc]
						if (cur_comb in prev_report_list): # single item that was misplaced 
							supposed, copy_prev, duplicate = obtain_supposed_loc(cur_comb, prev_report_list, copy_prev, duplicate, split_loc, new_report_list, cur_loc)
							item_keys[supposed] = cur_comb

					split_loc = sorted(split_loc)
					#Find supposed position of "broken-down" elements
					new_report_list, item_keys = reposition(split_loc, new_report_list, prev_report_list, single_item_misplaced, item_keys)
					items = item_keys.items()
					sorted_items = sorted(items)
					assert len(item_keys) == len(report_list)
					
					new_report_list = []
					for k,v in sorted_items:
						new_report_list.append(v)
					report_list = new_report_list
			prev_report_list = report_list

				
			#Traverse through each component/ segment and process the start/end points (starting with 1 since [CLS] and [SEP] are taken into consideration for location), account for subwords
			# Produce tuple (start_point, end_point) for each segment
			occ_dict={}	
			occ_freq={}
			prev_freq={}
			last_end_point = -1
			for counter in range (len(report_list)):
				if (isinstance(report_list[counter], int)):
					
					extracted_token = int(report_list[counter])
					if (extracted_token not in occ_freq):
						occ_freq[extracted_token] = 0
					else:
						occ_freq[extracted_token] += 1

					start_point=handle_start_point(id_dict, occ_freq,prev_freq, extracted_token, last_end_point)

					end_point = adjust_end_point_single(subword_mapping, start_point)
					#Avoid similar tokens at start_point and end_point
					if (end_token not in occ_freq):
						occ_freq[end_token] = 0
					else:
						occ_freq[end_token] += 1

					cur_point = (start_point, end_point)
					last_end_point = end_point
				elif (len(report_list[counter]) == 1):	
					extracted_token = report_list[counter][0]
					
					if (extracted_token not in occ_freq):
						occ_freq[extracted_token] = 0
					else:
						occ_freq[extracted_token] += 1

					start_point =  handle_start_point(id_dict, occ_freq,prev_freq, extracted_token, last_end_point)

					end_point = adjust_end_point_single(subword_mapping, start_point)
					cur_point = (start_point, end_point)
					last_end_point = end_point
				else:
				
					start_token = report_list[counter][0]
					end_token = report_list[counter][-1]	
				

					if (start_token not in occ_freq):
						occ_freq[start_token] = 0
					else:
						occ_freq[start_token] += 1
	
					inner_tokens = report_list[counter][1:-1]
					inner_occ_freq = {}
					for t in inner_tokens:	
						if (t not in inner_occ_freq):
							inner_occ_freq[t] = 1
						else:
							inner_occ_freq[t] += 1

					start_point =  handle_start_point(id_dict, occ_freq,prev_freq, start_token, last_end_point)
					#Avoid similar tokens at start_point and end_point
					if (end_token not in occ_freq):
						occ_freq[end_token] = 0
					else:
						occ_freq[end_token] += 1

					if (end_token in inner_occ_freq):
						if ((len(list(prev_freq.keys()))>=1) and (end_token in prev_freq)):
							end_point = id_dict[end_token][occ_freq[end_token]+ inner_occ_freq[end_token] + prev_freq[end_token]]	
						else:
							end_point = id_dict[end_token][occ_freq[end_token]+ inner_occ_freq[end_token]]		
					else:
						if (len(list(prev_freq.keys()))>=1 and end_token in prev_freq):
							end_point = id_dict[end_token][occ_freq[end_token] + prev_freq[end_token]]	
						else:
							end_point = id_dict[end_token][occ_freq[end_token]]	

					#Update end point if it is a part of subwords
					new_end_point = adjust_end_point_single(subword_mapping, end_point)
					last_end_point = new_end_point
					#Update prev_freq					
					for t in inner_occ_freq.keys():
						if (t not in prev_freq):
							prev_freq[t] = inner_occ_freq[t]
						else:
							prev_freq[t] += inner_occ_freq[t]
					cur_point = (start_point, new_end_point)


				#Obtain cut_score ( i.e. 5 segments, 4 cut_scores)	
				if (counter != len(report_list) - 1):
					score= calculate_cut_score(end_point,scores, new_score)
					cut_scores.append(score)
				current_level.append(cur_point)
		# Sanity check
		if (len(current_level) > 1): # MULTIPLE SEGMENTS on the same level
			prev_seg = None
			for item in current_level:
				cur_seg = item
				if (prev_seg != None):
					assert cur_seg[0] > prev_seg[1]
				prev_seg = cur_seg	
		save_info = [current_level, cut_scores]
		return_dict[i] = save_info
		if (i == desired_lev):
			break
	return return_dict, new_tree_depth, desired_lev

def right_branching(sent):
    if type(sent) is not list:
        return sent
    if len(sent) == 1:
        return sent[0]
    else:
        return [sent[0], right_branching(sent[1:])]


def left_branching(sent):
    if type(sent) is not list:
        return sent
    if len(sent) == 1:
        return sent[0]
    else:
        return [ left_branching(sent[:-1]), sent[-1]]


def create_level_dep(new_tree, level, id_dict, subwords, subword_mapping, tokenized_text, indexed_tokens, scores,num_occ_dict, new_score):
	if (len(new_tree) == 1):
		new_tree_depth = 0
	else:	
		new_tree_depth = getdepth(new_tree)
	return_level = None
	
	return_dict ={}
	prev_report_list = None
	# For each level
	for i in range (new_tree_depth + 1):
		extracted_level = i
		current_level = []

		cut_scores = []
		return_list = getrep_id(extracted_level,new_tree, new_tree_depth)
		if (extracted_level == 0):
			report_list = []
			report_list = removeNestings(return_list, report_list)
			start_point = id_dict[report_list[0]][0]
			end_point = id_dict[report_list[-1]][-1]

			val_at_end_point = subword_mapping[end_point]	
			val_in_mapping, val_in_count = np.unique(np.array(subword_mapping), return_counts=True)
			
			subword_idx  = val_in_count[list(val_in_mapping).index(val_at_end_point)].item()
			if (subword_idx > 1):
				new_end_point = end_point + (subword_idx-1)
			else:
				new_end_point = end_point
			current_level = [(start_point, new_end_point)]
		else:
			if (extracted_level != new_tree_depth):
				report_list = []
				counter = 0
				for e in return_list:
					if (e ==101 or e==102):
						report_list.append(e)
					else:
						r = []
						r = removeNestings_id(e, r)
						report_list.append(r)
					counter += 1
			else:
				report_list = return_list
			
			if (extracted_level == new_tree_depth):
				pass
			else:
				copy_id_dict = copy.deepcopy(id_dict)
				new_report_list= [0]*len(report_list)
				need_sort=[]
				locs = []
				prev_mismatch=[]
				expansion_locs =[]
				split = False
				if (prev_report_list !=None):
					for index in range (len(report_list)):
						if (index < len(prev_report_list) and split == False):	
							if (report_list[index] == prev_report_list[index]):

								need_sort.append(report_list[index])

								locs.append(index)		
								new_report_list[index] = report_list[index]
							else:
								locs.append(index)		
								need_sort.append(report_list[index])
								prev_mismatch.append(prev_report_list[index])
								expansion_locs.append(index)
								split=True
						else: # Final index
							if (index < len(prev_report_list)):
								prev_mismatch.append(prev_report_list[index])
							locs.append(index)
							need_sort.append(report_list[index])
				
					need_sort = sorted(need_sort, key=lambda x: get_key(x,copy_id_dict)) # rearranging segments by the first token location in a sentence
					split_loc = []
					duplicate_log = []
					s_index = 0
					single_item_misplaced =set()
					copy_prev_list = copy.deepcopy(prev_report_list)

					item_keys={}
					while (s_index <= len(need_sort)-1):
						if (need_sort[s_index]  in copy_prev_list): # single item		
							single_item_misplaced.add(locs[s_index])
							copy_prev_list.remove(need_sort[s_index])

						for j_index in range (s_index,len(need_sort)):
							combined_item = need_sort[s_index] + need_sort[j_index]

							reverse_combined_item = need_sort[j_index] + need_sort[s_index]
							if (combined_item in prev_report_list):
								if ((combined_item not in duplicate_log) or (combined_item in duplicate_log and prev_report_list.count(combined_item) > 1)):
									tmp1 = need_sort[s_index+1]
									need_sort[s_index+1] = need_sort[j_index]
									need_sort[j_index] = tmp1
									split_loc.append(locs[s_index])
									duplicate_log.append(combined_item)
									if (j_index not in single_item_misplaced):
										single_item_misplaced.add(j_index)

									s_index+= 1
									break
							elif (reverse_combined_item in prev_report_list):

								if ((reverse_combined_item not in duplicate_log) or (reverse_combined_item in duplicate_log and prev_report_list.count(reverse_combined_item) > 1)):
									tmp = need_sort[j_index]
									need_sort[j_index] = need_sort[s_index]
									need_sort[s_index] = tmp
									
									tmp1 = need_sort[s_index+1]
									need_sort[s_index+1] = need_sort[j_index]
									need_sort[j_index] = tmp1

									duplicate_log.append(reverse_combined_item)
									split_loc.append(locs[s_index])


									if (j_index not in single_item_misplaced):
										single_item_misplaced.add(j_index)
									s_index+= 1
									break
						s_index+=1		
					single_item_misplaced = list(single_item_misplaced)

					for l_index in range (len(locs)):
						new_report_list[locs[l_index]] = need_sort[l_index]

					
					# Find supposed position of single element
					split_loc = sorted(split_loc)
					duplicate=[]
					copy_prev = copy.deepcopy(prev_report_list)
					for idx in range (len(single_item_misplaced)): # Each element is a single element (not as a pair with anyting before or after)
						cur_loc = single_item_misplaced[idx]

						cur_comb = new_report_list[cur_loc]
						if (cur_comb in prev_report_list): # single item that was misplaced 
							supposed, copy_prev, duplicate = obtain_supposed_loc(cur_comb, prev_report_list, copy_prev, duplicate, split_loc, new_report_list, cur_loc)
							item_keys[supposed] = cur_comb

					split_loc = sorted(split_loc)
					#Find supposed position of "broken-down" elements
					new_report_list, item_keys = reposition(split_loc, new_report_list, prev_report_list, single_item_misplaced, item_keys)
					items = item_keys.items()
					sorted_items = sorted(items)

					assert len(item_keys) == len(report_list)
					
					new_report_list = []
					for k,v in sorted_items:
						new_report_list.append(v)
					report_list = new_report_list
			prev_report_list = report_list

				
			#Traverse through each component/ segment and process the start/end points (starting with 1 since [CLS] and [SEP] are taken into consideration for location), account for subwords
			# Produce tuple (start_point, end_point) for each segment
			occ_dict={}	
			occ_freq={}
			prev_freq={}
			last_end_point = -1
			for counter in range (len(report_list)):
				if (isinstance(report_list[counter], int)):
					
					extracted_token = int(report_list[counter])
					if (extracted_token not in occ_freq):
						occ_freq[extracted_token] = 0
					else:
						occ_freq[extracted_token] += 1

					start_point=handle_start_point(id_dict, occ_freq,prev_freq, extracted_token, last_end_point)
					end_point = start_point # no adjustment for dep tree

					#Avoid similar tokens at start_point and end_point
					if (end_token not in occ_freq):
						occ_freq[end_token] = 0
					else:
						occ_freq[end_token] += 1

					cur_point = (start_point, end_point)
					last_end_point = end_point
				elif (len(report_list[counter]) == 1):	
					extracted_token = report_list[counter][0]
					
					if (extracted_token not in occ_freq):
						occ_freq[extracted_token] = 0
					else:
						occ_freq[extracted_token] += 1

					start_point =  handle_start_point(id_dict, occ_freq,prev_freq, extracted_token, last_end_point)
					end_point = start_point
					cur_point = (start_point, end_point)
					last_end_point = end_point
				else:
				
					start_token = report_list[counter][0]
					end_token = report_list[counter][-1]	
				

					if (start_token not in occ_freq):
						occ_freq[start_token] = 0
					else:
						occ_freq[start_token] += 1
	
					inner_tokens = report_list[counter][1:-1]
					inner_occ_freq = {}
					for t in inner_tokens:	
						if (t not in inner_occ_freq):
							inner_occ_freq[t] = 1
						else:
							inner_occ_freq[t] += 1

					start_point =  handle_start_point(id_dict, occ_freq,prev_freq, start_token, last_end_point)
					#Avoid similar tokens at start_point and end_point
					if (end_token not in occ_freq):
						occ_freq[end_token] = 0
					else:
						occ_freq[end_token] += 1

					if (end_token in inner_occ_freq):
						if ((len(list(prev_freq.keys()))>=1) and (end_token in prev_freq)):
							end_point = id_dict[end_token][occ_freq[end_token]+ inner_occ_freq[end_token] + prev_freq[end_token]]	
						else:
							end_point = id_dict[end_token][occ_freq[end_token]+ inner_occ_freq[end_token]]		
					else:
						if (len(list(prev_freq.keys()))>=1 and end_token in prev_freq):
							end_point = id_dict[end_token][occ_freq[end_token] + prev_freq[end_token]]	
						else:
							end_point = id_dict[end_token][occ_freq[end_token]]	

					#Update end point if it is a part of subwords
					new_end_point = end_point # no subword adjustment for Dep Tree
					last_end_point = new_end_point
					#Update prev_freq					
					for t in inner_occ_freq.keys():
						if (t not in prev_freq):
							prev_freq[t] = inner_occ_freq[t]
						else:
							prev_freq[t] += inner_occ_freq[t]
					cur_point = (start_point, new_end_point)
				
				current_level.append(cur_point)

		# Sanity check
		if (len(current_level) > 1): # MULTIPLE SEG
			prev_seg = None
			for item in current_level:
				cur_seg = item
				if (prev_seg != None):
					assert cur_seg[0] > prev_seg[1]
				prev_seg = cur_seg	
		save_info = [current_level, cut_scores]
		return_dict[i] = save_info
	return return_dict, new_tree_depth
