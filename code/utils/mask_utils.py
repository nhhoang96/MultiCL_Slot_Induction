import numpy as np
import torch

# current_input_segment: [1,max_seg, max_seg_len,D]
# new_seg_len: [1, max_seg, 1]
# num_total_seg: # actual segs of the current input
# cur_index: Seg_idx (iterative)
# use_itself: decide to do Like_PM or Itself

#Example: I// like/book/ about/ garden
# Use Itself: [LIKE] = [[I]  + [LIKE]] + [[LIKE] + [BOOK]]
# Use LIke PM: [LIKE] = [I] + BOOK
def mask_adjacent(current_input_segment, new_seg_len, num_total_seg, cur_index, use_itself):
	#Less than 2 segments?
	if (num_total_seg ==1):
		cur_seg_rep = current_input_segment[:,cur_index:cur_index+1,:,:].sum(dim=-2) / (new_seg_len[:,cur_index:cur_index+1,:]+1e-20)
		print ("Seg 1", cur_seg_rep.shape)
	elif (num_total_seg == 2):
		if (cur_index == 0):
			if (use_itself == 'true'):	
				right_part = current_input_segment[:,0:num_total_seg,:,:] # [1,2,max_len,D]
				right_len = new_seg_len[:,0:num_total_seg,:]
				cur_seg_rep = right_part.sum(dim=-2) / (right_len + 1e-20)
				cur_seg_rep = cur_seg_rep.mean(dim=1,keepdim=True)
			else:

				right_part = current_input_segment[:,1:num_total_seg,:,:] # [1,2,max_len,D]
				right_len = new_seg_len[:,1:num_total_seg,:]
				cur_seg_rep = right_part.sum(dim=-2) / (right_len + 1e-20)
				cur_seg_rep = cur_seg_rep.mean(dim=1,keepdim=True)
		else:

			if (use_itself == 'true'):	
				left_part = current_input_segment[:,0:num_total_seg,:,:] # [1,2,max_len,D]
				left_len = new_seg_len[:,0:num_total_seg,:]
				cur_seg_rep = left_part.sum(dim=-2) / (left_len + 1e-20)
				cur_seg_rep = cur_seg_rep.mean(dim=1,keepdim=True)
			else:

				left_part = current_input_segment[:,0:num_total_seg-1,:,:] # [1,2,max_len,D]
				left_len = new_seg_len[:,0:num_total_seg -1,:]
				cur_seg_rep = left_part.sum(dim=-2) / (left_len + 1e-20)
				cur_seg_rep = cur_seg_rep.mean(dim=1,keepdim=True)
	else: # more than 2 segs
		if (cur_index == 0):
			if (use_itself == 'itself'):
				right_part = current_input_segment[:,0:2,:,:] # [1,2,max_len,D]
				right_len = new_seg_len[:,0:2,:]
				assert right_part.shape[1] == right_len.shape[1] == 2
			else:
				right_part = current_input_segment[:,0:1,:,:] #[1,1,max_len,D]
				right_len = new_seg_len[:,0:1,:]

				assert right_part.shape[1] == right_len.shape[1] == 1
			right_part = right_part.sum(dim=-2) / (right_len + 1e-20) #[1,2/1,D]
			cur_seg_rep = right_part.mean(dim=1, keepdim=True)
		elif (cur_index == num_total_seg -1):

			if (use_itself == 'itself'):
				left_part = current_input_segment[:,cur_index-1:cur_index+1,:,:] # [1,2,max_len,D]
				left_len = new_seg_len[:,cur_index-1:cur_index+1,:]
				assert left_part.shape[1] == left_len.shape[1] == 2
			else:
				left_part = current_input_segment[:,cur_index-1:cur_index,:,:] #[1,1,max_len,D]
				left_len = new_seg_len[:,cur_index-1:cur_index,:]

				assert left_part.shape[1] == left_len.shape[1] == 1
			left_part = left_part.sum(dim=-2) / (left_len + 1e-20) #[1,2/1,D]
			cur_seg_rep = left_part.mean(dim=1, keepdim=True)
		else:

			if (use_itself == 'itself'):
				left_part = current_input_segment[:,cur_index-1:cur_index+1,:,:] # [1,2,max_len,D]
				left_len = new_seg_len[:,cur_index-1:cur_index+1,:]
				assert left_part.shape[1] == left_len.shape[1] == 2
			else:
				left_part = current_input_segment[:,cur_index-1:cur_index,:,:] #[1,1,max_len,D]
				left_len = new_seg_len[:,cur_index-1:cur_index,:]
			
			if (use_itself == 'itself'):
				right_part = current_input_segment[:,cur_index-1:cur_index+1,:,:] # [1,2,max_len,D]
				right_len = new_seg_len[:,cur_index-1:cur_index+1,:]
			else:
				right_part = current_input_segment[:,cur_index-1:cur_index,:,:] #[1,1,max_len,D]
				right_len = new_seg_len[:,0:1,:]
			full_part = torch.cat([left_part, right_part],1)
			full_len = torch.cat([left_len, right_len],1)


			full_part = full_part.sum(dim=-2) / (full_len + 1e-20) #[1,2/1,D]
			cur_seg_rep = full_part.mean(dim=1, keepdim=True)
	assert len(cur_seg_rep.shape) == 3
	assert cur_seg_rep.shape[1] == 1

		
	return cur_seg_rep

#Example: I// like/book/ about/ garden
# Use Itself: [LIKE] = [[I]  + [LIKE]] + [[LIKE] + [BOOK] [ABOUT] [GARDEN]]
# Use LIke PM: [LIKE] = [I] + BOOK
def mask_all(current_input_segment, new_seg_len, num_total_seg, cur_index, use_itself):
	#Less than 2 segments?
	if (num_total_seg ==1):
		cur_seg_rep = current_input_segment[:,cur_index:cur_index+1,:,:].sum(dim=-2) / (new_seg_len[:,cur_index:cur_index+1,:]+1e-20)
		print ("Seg 1", cur_seg_rep.shape)
	elif (num_total_seg == 2):
		right_part = current_input_segment[:,0:num_total_seg,:,:] # [1,2,max_len,D]
		right_len = new_seg_len[:,0:num_total_seg,:]
		cur_seg_rep = right_part.sum(dim=-2) / (right_len + 1e-20)
		cur_seg_rep = cur_seg_rep.mean(dim=1,keepdim=True)
	else: # more than 2 segs
		if (cur_index == 0):
			if (use_itself == 'itself'):
				right_part = current_input_segment[:,:num_total_seg,:,:] # [1,2,max_len,D]
				right_len = new_seg_len[:,:num_total_seg,:]
				assert right_part.shape[1] == right_len.shape[1]
			else:
				right_part = current_input_segment[:,1:num_total_seg,:,:] #[1,1,max_len,D]
				right_len = new_seg_len[:,1:num_total_seg,:]

				assert right_part.shape[1] == right_len.shape[1]
			right_part = right_part.sum(dim=-2) / (right_len + 1e-20) #[1,2/1,D]
			cur_seg_rep = right_part.mean(dim=1, keepdim=True)

		elif (cur_index == num_total_seg -1):

			if (use_itself == 'itself'):
				left_part = current_input_segment[:,0:cur_index+1,:,:] # [1,2,max_len,D]
				left_len = new_seg_len[:,0:cur_index+1,:]
				assert left_part.shape[1] == left_len.shape[1]
			else:
				left_part = current_input_segment[:,0:cur_index,:,:] #[1,1,max_len,D] # not including the last token
				left_len = new_seg_len[:,0:cur_index,:]

				assert left_part.shape[1] == left_len.shape[1]
			left_part = left_part.sum(dim=-2) / (left_len + 1e-20) #[1,2/1,D]
			cur_seg_rep = left_part.mean(dim=1, keepdim=True)
		else:

			if (use_itself == 'itself'):
				left_part = current_input_segment[:,0:cur_index+1,:,:] # [1,2,max_len,D]
				left_len = new_seg_len[:,0:cur_index+1,:]
				assert left_part.shape[1] == left_len.shape[1] 
			else:
				left_part = current_input_segment[:,0:cur_index,:,:] #[1,1,max_len,D]
				left_len = new_seg_len[:,0:cur_index,:]	
				assert left_part.shape[1] == left_len.shape[1]

			if (use_itself == 'itself'):
				right_part = current_input_segment[:,cur_index:num_total_seg,:,:] # [1,2,max_len,D]
				right_len = new_seg_len[:,cur_index:num_total_seg,:]
				assert right_part.shape[1] == right_len.shape[1]
			else:
				right_part = current_input_segment[:,cur_index+1:num_total_seg,:,:] #[1,1,max_len,D]
				right_len = new_seg_len[:,cur_index+1: num_total_seg,:]
				assert right_part.shape[1] == right_len.shape[1]
			full_part = torch.cat([left_part, right_part],1)
			full_len = torch.cat([left_len, right_len],1)


			full_part = full_part.sum(dim=-2) / (full_len + 1e-20) #[1,2/1,D]
			cur_seg_rep = full_part.mean(dim=1, keepdim=True)
	assert len(cur_seg_rep.shape) == 3
	assert cur_seg_rep.shape[1] == 1

		
	return cur_seg_rep
	

def extract_phrase(label, pred, toks):
	label_phrases=[]
	inner_label = []
	pred_phrases=[]
	inner_pred=[]
	for i in range (len(label)):
		if (label[i] == 1): #ignore label == -1
			if (len(inner_label) == 0): # first item, then do both
				inner_label.append(toks[i])
				inner_label.append(toks[i+1])
			else:
				inner_label.append(toks[i+1])
		elif (label[i] == 0):
			if (len(inner_label) >= 1):
				phrase = ' '.join(inner_label) 
				label_phrases.append(phrase)
				inner_label = [] # reset for the new label_phrases in the sentence

		if ((i == len(label) -1) and (len(inner_label) > 0) and label[i] !=-1): # if tie until the end of the string
			phrase = ' '.join(inner_label)
			label_phrases.append(phrase)


		if (pred[i] == 1):
			if (len(inner_pred) == 0): # first item, then do both
				inner_pred.append(toks[i])
				inner_pred.append(toks[i+1])
			else:
				inner_pred.append(toks[i+1])
		elif (pred[i] == 0):
			if (len(inner_pred) >= 1):
				phrase = ' '.join(inner_pred) 
				pred_phrases.append(phrase)
				inner_pred = [] # reset for the new pred_phrases in the sentence

		if ((i == len(pred) -1) and (len(inner_pred) > 0) and pred[i] !=-1): # if tie until the end of the string
			phrase = ' '.join(inner_pred)
			pred_phrases.append(phrase)
	return label_phrases, pred_phrases


#---- Printing Debugging Info -------#
def print_text_info(total_tie_prediction, total_focused_tie_label, total_texts, total_pos_tie_prediction, total_pos_tie_label, total_pos_texts, total_seg_text, total_pos_seg_text, total_neg_seg_text, total_intent_label, all_pos_sim, all_neg_sim,all_logits, config):
	for idx in range (total_texts.shape[0]):
		pred = total_tie_prediction[idx]
		label = total_focused_tie_label[idx]
		text = total_texts[idx]
		toks = text.split(' ')
		label_phrases, pred_phrases = extract_phrase(label, pred, toks)
		if (config['model_type'] == 'cl'):
			pos_pred = total_pos_tie_prediction[idx]
			pos_label = total_pos_tie_label[idx]
			pos_text = total_pos_texts[idx]
			pos_toks = pos_text.split(' ')
			pos_label_phrases, pos_pred_phrases = extract_phrase(pos_label, pos_pred, pos_toks)
			
			ori_seg_text = total_seg_text[idx]
			pos_seg_text = total_pos_seg_text[idx]
			neg_seg_text = total_neg_seg_text[idx]
			pos_sim,neg_sim = all_pos_sim[idx], all_neg_sim[idx]

			print ("Test example (Golden-Predict-Text-Intent)", label_phrases, pred_phrases, text, ori_seg_text, total_intent_label[idx], all_logits[idx])
			print ("Pos (Golden-Predict-Text-Intent)", pos_label_phrases, pos_pred_phrases, pos_text, pos_seg_text, pos_sim.sum())
			print ("Neg (Golden-Predict-Text-Intent)", neg_seg_text, neg_sim.sum())
		else:
			print ("Test example (Golden-Predict-Text-Intent)", label_phrases, pred_phrases,  text, total_intent_label[idx])

		print ("\n")
def print_len_info(all_ori_seg_len, all_pos_seg_len, all_neg_seg_len, all_ori_seg_len_std, all_pos_seg_len_std, all_neg_seg_len_std, model_type):
        if (model_type == 'cl'):
        #if (config['use_cl'] == 'true'):
                all_ori_seg_len = np.array(all_ori_seg_len)
                all_pos_seg_len = np.array(all_pos_seg_len)
                all_neg_seg_len = np.array(all_neg_seg_len)

                all_ori_seg_len_std = np.array(all_ori_seg_len_std)
                all_pos_seg_len_std = np.array(all_pos_seg_len_std)
                all_neg_seg_len_std = np.array(all_neg_seg_len_std)
                print ("Original", np.mean(all_ori_seg_len), np.std(ori_seg_len))
                print ("Pos", np.mean(all_pos_seg_len), np.std(pos_seg_len))
                print ("Neg", np.mean(all_neg_seg_len), np.std(neg_seg_len))
                print ("\n")
                print ("Original STD", np.mean(all_ori_seg_len_std), np.std(ori_seg_len_std))
                print ("Pos STD", np.mean(all_pos_seg_len_std), np.std(pos_seg_len_std))
                print ("Neg STD", np.mean(all_neg_seg_len_std), np.std(neg_seg_len_std))

def save_train_info(ori_emb, seg_outp, seg_mask, seg_word_outp, seg_word_mask, seg_graph, seg_score, seg_dict, desired_lev):
	data_dict={}
	data_dict['ori'] = ori_emb #sub-token level #bsz, max_len,D
	data_dict['seg_outp'] = seg_outp
	data_dict['seg_word_outp'] = seg_word_outp
	data_dict['seg_mask'] = seg_mask
	data_dict['seg_word_mask'] = seg_word_mask
	data_dict['seg_graph'] = seg_graph
	data_dict['seg_score'] = seg_score
	data_dict['seg_dict'] = seg_dict
	data_dict['desired_lev'] = desired_lev

	return data_dict	
	
def return_train_info(data_dict, key):
	return data_dict[key]

def extract_full_outp(data_dict):
	return data_dict['seg_word_outp'], data_dict['seg_word_mask'], data_dict['seg_mask']

def extract_outp(data_dict):
	return data_dict['seg_outp'], data_dict['seg_mask']


def update_anchor_pos_neg_info(anchor_list, pos_list, neg_list, anchor_item_list, pos_item_list, neg_item_list):
	if (isinstance(anchor_item_list[0],list)): # Need to extend
		anchor_list.extend(anchor_item_list)
		pos_list.extend(pos_item_list)
		neg_list.extend(neg_item_list)
	else: #otherwise, append
		anchor_list.append(anchor_item_list)
		pos_list.append(pos_item_list)
		neg_list.append(neg_item_list)

	return anchor_list, pos_list, neg_list


def update_anchor_info(anchor_list, pos_list, neg_list, anchor_item_list, pos_item_list, neg_item_list):
	if (isinstance(anchor_item_list[0],list)): # Need to extend
		anchor_list.extend(anchor_item_list)
		pos_list.extend(pos_item_list)
		neg_list.extend(neg_item_list)
	else: #otherwise, append
		anchor_list.append(anchor_item_list)
		pos_list.append(pos_item_list)
		neg_list.append(neg_item_list)

	return anchor_list, pos_list, neg_list

def update_np_anchor_pos_neg_info(anchor_list, pos_list, neg_list, anchor_item_list, pos_item_list, neg_item_list):
	anchor_list = np.concatenate((anchor_list, anchor_item_list))
	pos_list = np.concatenate((pos_list, pos_item_list))
	neg_list = np.concatenate((neg_list, neg_item_list))
	return anchor_list, pos_list, neg_list
