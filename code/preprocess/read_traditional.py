import numpy as np
import gc
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import pandas as pd

from . import bert_segment, load_helper
from utils import bert_segment_utils, utils
import re
import os
data_prefix = '../../dataset/'

def tokenize_text(text, tokenizer):
                tokenized_text = tokenizer.tokenize(text)
                tokenized_text.insert(0, '[CLS]')
                tokenized_text.append('[SEP]')
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

                return tokenized_text, indexed_tokens

def load_bert_sentence(file_path, tokenizer, all_slots, all_intents, sc_dict, label_info, max_len, seg_levels, max_seg_level, adaptive_depth,max_slot_len):
        x_train, y_train, x_len, sent, mask, tokenized_texts = [],[],[],[],[],[]
        sub, full_mapping, old_subword_mapping = [], [],[]

        slot_labels = []
        focused_tie_break_labels = [] # for actual slot induction
        full_tie_break_labels = [] # typical phrasal segmentation
        #Label info
        label_toks, label_tok_masks, label_old_maps, label_subwords, label_tok_names, label_seg= [],[],[],[],[],[]

        f = open(file_path, 'r')
        for line in f:
                arr = line.strip().split('\t')
                label = [w for w in arr[0].split(' ')]
                cname = ' '.join(label)
                if not (cname in sc_dict):
                        continue		
                text =re.sub('\s\s+', ' ', arr[1])
                text = re.sub(r"[\([)\]']",'',text)
                text = text.replace("  ", " ")

                if (len(text.split(' ')) > 1): # ignore one token sentence
                        
                        tok_idx = tokenizer.encode(text, add_special_tokens=True)
                        tokenized_text, ind_tokenized_text = tokenize_text(text, tokenizer)
                        mapping = bert_segment_utils.rematch(text, tokenized_text)
                        
                        
                        old_mapping = bert_segment_utils.match_tokenized_to_untokenized(tokenized_text, text.split(' '))

                        if not (type(old_mapping) == bool): # need to meet condition of PM processing
                                tokenized_texts.append(tokenized_text)
                                subwords=[]
                                for i in range(1,len(tokenized_text)):
                                        if (old_mapping[i] == old_mapping[i-1]):
                                                subwords.append(i)

                                old_subword_mapping.append(old_mapping)
                                sub.append(subwords)
                                text_len = len(text.split(' ')) # based on WORD, NOT SUBWORD
                                label_toks.append(label_info[sc_dict[cname]]['tok_id'])
                                label_tok_masks.append(label_info[sc_dict[cname]]['tok_mask'])
                                label_subwords.append(label_info[sc_dict[cname]]['tok_subwords'])
                                label_old_maps.append(label_info[sc_dict[cname]]['tok_old_mapping'])
                                        
                                label_tok_names.append(label_info[sc_dict[cname]]['tok_name'])

                                x_len.append(text_len)
                                y_train.append(sc_dict[cname])
                                sent.append (text)
                                full_mapping.append(mapping)
                                tmp, current_mask = utils.create_mask(ind_tokenized_text, max_len, ind_tokenized_text)	
                                
                                x_train.append(tmp) 
                                mask.append(current_mask)
                                slot = arr[-1].strip()
                                slot_elements = slot.split(' ')
                                max_slot_len = max(max_slot_len, len(slot_elements)) 

                                
                                focused_tie_break, full_tie_break = utils.convert_slot_to_tie_break(slot_elements, idx2slot=None, slot_set=all_slots)
                                focused_tie_break_labels.append(focused_tie_break)
                                full_tie_break_labels.append(full_tie_break)
                                slot_labels.append(slot)

                        else:
                                pass # Error case, ignore this sample
                else:
                        print ("Discard len < 1")

        f.close()
        gc.collect()
        x_train, y_train, x_len = np.asarray(x_train), np.array(y_train), np.array(x_len)
        mask = np.array(mask)
        tokenized_texts = np.array(tokenized_texts, dtype='object')
        full_mapping = np.array(full_mapping, dtype='object')
        slots = np.array(slot_labels, dtype='object')

        focused_tie_break_labels = np.array(focused_tie_break_labels, dtype='object')
        full_tie_break_labels = np.array(full_tie_break_labels, dtype='object')

        sub = np.array(sub, dtype='object')

        old_subword_mapping = np.array(old_subword_mapping, dtype='object')
        label_toks = np.array(label_toks)
        label_tok_masks = np.array(label_tok_masks)
        label_old_maps = np.array(label_old_maps, dtype='object')
        label_subwords = np.array(label_subwords, dtype='object')
        label_tok_names = np.array(label_tok_names, dtype='object')


        return x_train, y_train, x_len, np.array(sent, dtype='<U2000'), mask, tokenized_texts,slots,full_mapping, max_slot_len, focused_tie_break_labels, full_tie_break_labels,sub, old_subword_mapping, label_toks, label_tok_masks, label_old_maps, label_subwords, label_tok_names


def save_data(data_type, x,y,in_len,text, mask,tok_text, slot, mapping, focused_tie_break_label, full_tie_break_label, subword,old_subword, label_tok, label_mask,label_old_maps, label_subwords, label_tok_names, data):
		data['x_' + data_type] = x
		data['y_' + data_type] = y
		data['len_' + data_type] = in_len
		data['text_' + data_type] = text
		data['mask_' + data_type] = mask
		#data['seg_' + data_type] = seg_dict

		#data['seg_depth_' + data_type] = seg_depth
		data['tok_text_' + data_type] = tok_text

		data['slot_' + data_type] = slot

		data['mapping_' + data_type] = mapping
		if (focused_tie_break_label is not None):
			data['focused_tie_break_label_' + data_type] = focused_tie_break_label
			data['full_tie_break_label_' + data_type] = full_tie_break_label
		else:

			data['focused_tie_break_label_' + data_type] = None
			data['full_tie_break_label_' + data_type] = None
		data['subword_' + data_type] = subword

		data['old_subword_mapping_' + data_type] = old_subword
		data['label_tok_id_' + data_type] = label_tok
		data['label_tok_mask_' + data_type] = label_mask

		data['label_tok_name_' + data_type] = label_tok_names
		data['label_subword_' + data_type] = label_subwords	
		data['label_old_subword_mapping_' + data_type] = label_old_maps
		return data



def convert_to_idx(slot_names, slot_set, max_slot_len):
		all_slot_idx = []
		for index in range (slot_names.shape[0]):
			cur_slot = slot_names[index]
			slot_elements = cur_slot.split(' ')
			slot_idx= []
			for s in slot_elements:
				slot_idx.append(slot_set.index(s))
			
			all_slot_idx.append(slot_idx)
		
		all_slot_idx = np.array(all_slot_idx, dtype='object')
		return all_slot_idx


def process_label(label_dict, tokenizer, max_len):
	label_names = (label_dict.keys())
	labels={}
	counter = 0
	for n in label_names:
		tok_label, tok_label_id = tokenize_text(n, tokenizer)
		old_mapping = bert_segment_utils.match_tokenized_to_untokenized(tok_label, n.split(' '))

		subwords=[]
		for i in range(1,len(tok_label)):
			if (old_mapping[i] == old_mapping[i-1]):
				subwords.append(i)
		tok_mask = [1] * len(tok_label_id)

		#Pad tok_id
		if (len(tok_label_id) < max_len):
			zeros = [0]*(max_len - len(tok_label_id))
			tok_label_id.extend(zeros)
			tok_mask.extend(zeros)

		info={}
		info['name'] = n
		info['tok_name'] = tok_label
		info['tok_id'] = tok_label_id
		info['tok_mask'] = tok_mask
		info['tok_old_mapping'] = old_mapping
		info['tok_subwords'] = subwords
		labels[counter] = info
		counter += 1
	return labels

def load_bert(training_data_path, valid_data_path, test_data_path, data,label_dict, seg_level, adaptive_depth, dataset):
        print ("------------------load BERT begin-------------------")
        model_dir = '../bert_model/'
        tokenizer = BertTokenizer.from_pretrained(model_dir)

        print ("------------------load BERT end-------------------")
        all_paths = [training_data_path, valid_data_path, test_data_path]


        all_intents, all_slots = utils.get_all_intent_slot_labels(all_paths)
        max_len=64
        max_seg=50

        label_info = process_label(label_dict, tokenizer,max_len)
        max_slot_len = 0

        print ("----- Load training ------")
        x_tr, y_tr, x_len_tr,tr_sent, tr_mask, tok_text_tr, slots_name_tr, mapping_tr,max_slot_len,focused_tie_break_label_tr, full_tie_break_label_tr, subword_tr,old_subword_tr,label_tok_tr, label_mask_tr, label_old_maps_tr, label_subwords_tr, label_tok_names_tr = load_bert_sentence(training_data_path, tokenizer, all_slots,all_intents, label_dict, label_info, max_len, seg_level, max_seg, adaptive_depth, max_slot_len)
        gc.collect()

        print ("------ Load eval --------")
        x_valid, y_valid, x_len_valid, valid_sent,valid_mask, tok_text_valid,slots_name_valid, mapping_valid, max_slot_len, focused_tie_break_label_valid, full_tie_break_label_valid, subword_valid,old_subword_valid, label_tok_valid, label_mask_valid, label_old_maps_valid, label_subwords_valid, label_tok_names_valid = load_bert_sentence(valid_data_path, tokenizer, all_slots, all_intents, label_dict, label_info, max_len,seg_level, max_seg, adaptive_depth,max_slot_len)

        print ("-----Load test---------")
        x_te, y_te, x_len_te, te_sent,te_mask, tok_text_te,slots_name_te, mapping_te, max_slot_len, focused_tie_break_label_te,full_tie_break_label_te, subword_te,old_subword_te, label_tok_te, label_mask_te, label_old_maps_te, label_subwords_te, label_tok_names_te = load_bert_sentence(test_data_path, tokenizer, all_slots, all_intents, label_dict, label_info, max_len,seg_level, max_seg, adaptive_depth, max_slot_len)

        distinct_slot=[]
        for s in all_slots:
                if (s != 'O'):
                        s = s.split('-')[-1]
                        if (s not in distinct_slot):
                                distinct_slot.append(s)
                else:
                        distinct_slot.append(s)

        slots_tr = convert_to_idx(slots_name_tr, all_slots, max_slot_len)
        slots_te = convert_to_idx(slots_name_te, all_slots, max_slot_len)
        slots_valid = convert_to_idx(slots_name_valid, all_slots, max_slot_len)

        data = save_data('tr', x_tr,y_tr,x_len_tr, tr_sent, tr_mask, tok_text_tr,slots_tr, mapping_tr, focused_tie_break_label_tr, full_tie_break_label_tr, subword_tr,old_subword_tr,label_tok_tr, label_mask_tr, label_old_maps_tr, label_subwords_tr, label_tok_names_tr, data)				 

        data = save_data('te', x_te,y_te,x_len_te, te_sent, te_mask, tok_text_te,slots_te,mapping_te, focused_tie_break_label_te, full_tie_break_label_te, subword_te, old_subword_te, label_tok_te, label_mask_te,label_old_maps_te, label_subwords_te, label_tok_names_te, data)

        data = save_data('val', x_valid,y_valid,x_len_valid, valid_sent, valid_mask, tok_text_valid,slots_valid,mapping_valid,focused_tie_break_label_valid, full_tie_break_label_valid, subword_valid, old_subword_valid, label_tok_valid, label_mask_valid,label_old_maps_valid, label_subwords_valid, label_tok_names_valid, data)

        data['all_intents'] = all_intents
        data['all_slots'] = all_slots
        data['max_len'] = max_len
        data['max_seg'] = max_seg
        data['distinct_slot'] = distinct_slot
        return data

def obtain_label_dict(labels):
	#labels = np.array(df_train.iloc[:,0])
	unique_values = np.unique(labels)
	label_dict={}
	for j in range (unique_values.shape[0]):
		label_dict[unique_values[j]] = j

	return label_dict


def read_datasets(dataset, seg_level, adaptive_depth=False, retrain=False):

                print ("------------------read datasets begin-------------------")
                data = {}
                if ("SNIPS" in dataset):		
                                data_path = data_prefix + dataset + '/'
                elif ("ATIS" in dataset):

                                data_path = data_prefix + dataset + '/'

                training_data_path = data_path + 'train/text'
                test_data_path = data_path +'/test/text'
                valid_data_path = data_path +'/valid/text'
                if (retrain == True):	
                        training_data_path = data_prefix + dataset+ '_P1/train/text'
                        #training_data_path = data_prefix + dataset+ '_P2/text'
                        test_data_path = data_prefix + dataset +'_P2/text'

                        #valid_data_path = data_prefix + dataset +'_P2/text'
                        valid_data_path = data_prefix + dataset +'_P1/valid/text'
                df_train = np.array(pd.read_csv(training_data_path, sep='\t',usecols=[0]))

                df_test = np.array(pd.read_csv(test_data_path, sep='\t', usecols=[0]))

                full_df = np.concatenate((df_train, df_test),0)
                full_df = full_df.reshape(-1,)
                label_dict = obtain_label_dict(full_df)	
                data = load_bert(training_data_path, valid_data_path, test_data_path, data,label_dict, seg_level,adaptive_depth,dataset)
                print ("------------------read datasets end---------------------")
                return data
