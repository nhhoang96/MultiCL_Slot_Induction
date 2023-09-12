from transformers import*
import numpy as np
import torch
import torch.nn as nn
import unicodedata
import re
import copy

from utils import bert_segment_utils

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
LAYER = int(13)

#--- Impact Matrix Helper function	----- #
def prepare_tensor(mapping, indexed_tokens,tokenized_text, current_index):
	# Mask current token of interest
	id_for_all_i_tokens = get_all_subword_id(mapping, current_index) # get id for the current interested token
	tmp_indexed_tokens = list(indexed_tokens)
	for tmp_id in id_for_all_i_tokens:
			if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
					tmp_indexed_tokens[tmp_id] = mask_id

	# Mask the interaction with any other tokens
	one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
	for j in range(0, len(tokenized_text)):
		id_for_all_j_tokens = get_all_subword_id(mapping, j)
		for tmp_id in id_for_all_j_tokens:
			if mapping[tmp_id] != -1:
				one_batch[j][tmp_id] = mask_id
	# 2. Convert one batch to PyTorch tensors
	tokens_tensor = torch.tensor(one_batch)
	segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
	if (torch.cuda.is_available()):
		tokens_tensor = tokens_tensor.cuda()
		segments_tensor = segments_tensor.cuda()
	return tokens_tensor, segments_tensor


def create_single_layer_matrix(tokenized_text, indexed_tokens, mapping, subword_mapping, model, tgt_text, indexed_tgt, tgt_mapping, mask_id, subwords, level):
	matrix = []
	# 1. Generate mask indices
	for i in range(0, len(tokenized_text)): # each sentence
		tokens_tensor, segments_tensor = prepare_tensor(mapping, indexed_tokens,tokenized_text, i)
		if (torch.cuda.is_available()):
			model.cuda()
	
		# 3. get only the last hidden state
		with torch.no_grad():
			model_outputs = model(tokens_tensor, segments_tensor)
			last_hidden_states = model_outputs[-1][-1] # last hidden layer
		# 4. get hidden states for word_i in one batch
			hidden_states_for_token_i = last_hidden_states[:,i,:].cpu().numpy()	# Last hidden state
			matrix.append(hidden_states_for_token_i)
	init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
	for i, hidden_states in enumerate(matrix):
		base_state = hidden_states[i]
		for j, state in enumerate(hidden_states):
			init_matrix[i][j] = np.linalg.norm(base_state - state)
		
	id_dict={}
	num_occ_dict={}
	for i in range (len(indexed_tokens)):
		if not (indexed_tokens[i] in id_dict):
			id_dict[indexed_tokens[i]] = [i]
			num_occ_dict[indexed_tokens[i]]=1
		else:
			temp = id_dict[indexed_tokens[i]] 
			if (isinstance(temp, list)):
				temp.append(i)
			else:
				temp = [temp]
				temp.append(i)
			id_dict[indexed_tokens[i]] = temp
			num_occ_dict[indexed_tokens[i]] +=1

	init_matrix, simplified_text, simplified_toks = bert_segment_utils.cleanup(init_matrix, tokenized_text, indexed_tokens, subwords) # Remove subwords
	new_tree,score_split = bert_segment_utils.mart(init_matrix, simplified_toks, 'false')	
	if (isinstance(new_tree, int)):
		new_tree = [new_tree]
	
	return_level,_ = bert_segment_utils.create_level_dict(new_tree, level, id_dict,subword_mapping, mapping, tokenized_text,indexed_tokens, init_matrix, num_occ_dict,'false')
	return new_tree, return_level


def get_depth_stats(tokenized_text, indexed_tokens, mapping, model, tgt_text, indexed_tgt, tgt_mapping, mask_id, subwords, level):
	matrix = []
	# 1. Generate mask indices
	for i in range(0, len(tokenized_text)):
		id_for_all_i_tokens = get_all_subword_id(mapping, i)
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
		if (torch.cuda.is_available()):
			tokens_tensor = tokens_tensor.cuda()
			segments_tensor = segments_tensor.cuda()
			model.cuda()

		# 3. get only the last hidden state
		with torch.no_grad():
			model_outputs = model(tokens_tensor, segments_tensor)
			last_hidden_states = model_outputs[-1][-1] # last hidden layer

		# 4. get hidden states for word_i in one batch
			hidden_states_for_token_i = last_hidden_states[:,i,:].cpu().numpy()	# Last hidden state
			matrix.append(hidden_states_for_token_i)

	init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
	for i, hidden_states in enumerate(matrix):
		base_state = hidden_states[i]
		for j, state in enumerate(hidden_states):
			init_matrix[i][j] = np.linalg.norm(base_state - state)
		
	init_matrix, simplified_text, simplified_toks = cleanup(init_matrix, tokenized_text, indexed_tokens, subwords)
	new_tree,_ = mart(init_matrix, simplified_toks, 'false')	
	if (isinstance(new_tree, int)):
		new_tree = [new_tree]
	
	depth = getdepth(new_tree)
	return depth

def create_subwords(tokenized_text):
	subwords =[]
	for i in range (len(tokenized_text)):
		if (tokenized_text[i].startswith('##')):
			subwords.append(i)
	return subwords
		
def tokenize_text(text, tokenizer):
	tokenized_text = tokenizer.tokenize(text)
	#tokenized_tgt = tokenizer.tokenize(tgt_text)
	tokenized_text.insert(0, '[CLS]')
	tokenized_text.append('[SEP]')
	# Convert token to vocabulary indices
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	return tokenized_text, indexed_tokens

def get_all_subword_id(mapping, idx):
	current_id = mapping[idx]
	id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
	return id_for_all_subwords


def getdepth(tree):
	if not(isinstance(tree, list)):
		return 0
	leftdepth = getdepth(tree[0])
	rightdepth = getdepth(tree[1])
	if (leftdepth > rightdepth):
		return leftdepth + 1
	else:
		return rightdepth + 1

# Only printing for testing purposes
def print_test(text, tokenized_text, tree,return_level):
	print ("Original text", text)
	print ("Tokenized text", tokenized_text)
	print ("Parsed tree", tree)
	print ("Current segment level", return_level)
	print ("\n")



#------ Main Functions --------------------#
def build_impact_matrix(batch_text, batch_intent_label, batch_slot_label, level):
	batch_return = []	
	counter = 0	
	return_slot_label=[]

	return_intent_label=[]
	return_text = []
	print ("---BERT segment started ----")
	percent_data = round(0.4 * batch_text.shape[0])
	for i in range (batch_text.shape[0]):
		text = batch_text[i]
		tokenized_text, indexed_tokens = tokenize_text(text, tokenizer)
		mapping = bert_segment_utils.match_tokenized_to_untokenized(tokenized_text, text.split(' '))		
		if (len(text.split(' ')) > 1 and not type(mapping) == bool):
			# Start tokenized text	
			subwords = create_subwords(tokenized_text)

			subword_mapping = bert_segment_utils.rematch(text, tokenized_text)

			#subword_mapping = bert_segment_utils.match_tokenized_to_untokenized(tokenized_text,text)
			tree, return_level = create_single_layer_matrix(tokenized_text, indexed_tokens, mapping, subword_mapping, model, None,None,None, mask_id, subwords, level)
			print_test(text, tokenized_text, tree,return_level)
			batch_return.append(return_level)
			return_slot_label.append(batch_slot_label[i])
			return_intent_label.append(batch_intent_label[i])
			return_text.append(text)
			counter += 1
			#ratio = counter / batch_text.shape[0]
			if (counter % percent_data == 0):
				print ("---Processed ", round((counter / batch_text.shape[0]) * 100 ,2), "percent data ---")
	print ("---BERT segment ended ----")
	return batch_return, return_text, return_intent_label, return_slot_label
 
#def obtain_depth(batch_x, batch_text, level):
#	batch_return = []
#	for i in range (batch_x.shape[0]):
#		cur_dict={}
#		cur_x = batch_x[i]
#		text = batch_text[i]
#		# Start tokenized text	
#		tokenized_text, indexed_tokens = tokenize_text(text, tokenizer)
#		mapping = bert_segment_utils.match_tokenized_to_untokenized(tokenized_text, text.split(' '))		
#		subwords = create_subwords(tokenized_text)
#		depth = get_depth_stats(tokenized_text, indexed_tokens, mapping, model, None,None,None, mask_id, subwords, level)
#		batch_return.append(depth)
#				
#	return batch_return 


