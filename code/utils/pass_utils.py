import numpy as np

def save_train_info(ori_emb, seg_outp, seg_mask, seg_word_outp, seg_word_mask, seg_graph, seg_score, seg_dict, desired_lev,rand_full_outp=None, rand_full_mask=None):
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
	data_dict['rand_full_outp'] = rand_full_outp
	data_dict['rand_full_mask'] = rand_full_mask
	return data_dict	
	
def return_train_info(data_dict, key):
	return data_dict[key]

def extract_full_outp(data_dict):
	return data_dict['seg_word_outp'], data_dict['seg_word_mask'], data_dict['seg_mask'], data_dict['rand_full_outp'], data_dict['rand_full_mask']

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


def update_np_info(example_info,info_dict, type_data): #type_data=['pos', 'neg'] 
        all_keys = list(info_dict.keys())
        src_keys = list(example_info[type_data].keys())	
        for k in all_keys:
                if (k != 'seg_text'):
                        try:
                            assign_info = np.array(info_dict[k])
                        except:
                            assign_info = np.array(info_dict[k], dtype='object')
                        if (k not in src_keys):
                                example_info[type_data][k] = assign_info
                        else:
                                example_info[type_data][k] = np.concatenate((example_info[type_data][k], assign_info))
                else: # Length of seg_text is not fixed

                        if (k not in src_keys):
                                temp_list = []

                                for j in range (len(info_dict[k])):
                                        temp_list.append(info_dict[k][j])
                                example_info[type_data][k] = temp_list
                        else:
                                for j in range (len(info_dict[k])):
                                        example_info[type_data][k].append(info_dict[k][j])

        return example_info

