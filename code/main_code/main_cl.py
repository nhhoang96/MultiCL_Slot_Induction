# -*- coding: utf-8 -*-
#-Generic Utils ---#
import os
import time
import copy
import argparse
import csv
import gc
import math

#Tensor related modules
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random


#---Helper Implementations
from preprocess import read_traditional
from model import cl_model
from utils import bert_segment_utils, pass_utils, cl_utils, log_utils, utils,  train_utils

def parse_argument():
	parser = argparse.ArgumentParser()

	#---- Experiment Settings ------#
	parser.add_argument("--num_epochs", type=int, default=1)
	parser.add_argument('--ckpt_dir', type=str, default = './ckpt/')
	parser.add_argument('--learning_rate', type=float, default=1e-5)
	parser.add_argument('--model_type', type=str, default='cl')
	parser.add_argument('--num_run', type=int, default=1)
	parser.add_argument('--dataset', type=str, default='SNIPS_MOD')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument("--cur_seed", type=int, default=1)
	parser.add_argument('--seg_level',nargs='+', type=int, default=-1)
	parser.add_argument('--test_eff', type=str, default='true') #implement efficient PM

	#---- Contrastive Learning Settings------#
	parser.add_argument('--use_attn_reward', type=str, default='false')
	parser.add_argument('--use_relu',type=str, default='false')

	parser.add_argument('--use_seg',type=str, default='true')
	#---Augmentation Details ----#
	parser.add_argument('--mask_type',type=str,default='word_lr')
	parser.add_argument('--augment_pos', type=str, default='false') #only mask positive (not negative)

	parser.add_argument('--use_ceil', type=str, default='true')
	parser.add_argument('--ratio_seg', type=float, default='0.15')
	parser.add_argument('--num_seg', type=int, default='1')
	parser.add_argument('--fixed_seg', type=str, default='false')
	parser.add_argument('--augment', type=str, default='true')
	parser.add_argument('--level_aug', type=str, default='seg') # for word_lr augmentation type
	parser.add_argument('--mask_word_side',type=str,default='itself') #word_lr: lr/itself/all
	parser.add_argument('--ratio_word_in_seg', type=float, default=0.0) #word_lr (on word_level)
	parser.add_argument('--word_in_seg_ceil', type=str, default='false')


	parser.add_argument('--rand_seg', type=str, default='false') #word_lr: on word_level, masking on random seg, instead of ALL segs
	parser.add_argument('--all_left_right', type=str, default='false') #for mask_itself/like_pm: false: only adjacent, true: all left/right of current seg
	parser.add_argument('--rand_mask_itself', type=str, default='false') #random_masking with like_pm/itself

	parser.add_argument('--percent_depth', type=float, default='-1')
	parser.add_argument('--new_score',type=str,default='false') #changing denominator
	parser.add_argument('--score_version',type=str, default='original') #more versions of score

	parser.add_argument('--norm_score', type=str, default='true')
	parser.add_argument('--num_neg_samples', type=int, default=1)

	parser.add_argument('--test_mode', type=str, default='false')
	parser.add_argument('--naive_meanpool', type=str, default='false')

	#---Self supervision ----#	
	parser.add_argument('--self_supervise', type=str, default='false')
	parser.add_argument('--tune',type=str, default='false')

	parser.add_argument('--write_log',type=str, default='true')
	parser.add_argument('--gpu_id',type=int, default=1)

	#Loss Coefficients
	parser.add_argument('--sent_coeff', type=float, default=1.0)
	parser.add_argument('--seg_coeff', type=float, default=1.0)
	parser.add_argument('--sent_temp', type=float, default=0.05)
	parser.add_argument('--seg_temp', type=float, default=0.05)

	args = parser.parse_args()
	config = args.__dict__
	return config

def load_model(config, seed_val,pretrain=False):

		if (config['model_type'] == 'cl'):
			model = cl_model.CL_Model(config)

		#train_params=[]
		#for n,p in model.named_parameters():
		#	if (p.requires_grad == True):
		#		train_params.append(n)
		#print ("Trainable Parameters:", train_params)

		optimizer= optim.Adam(model.parameters(), lr=config['learning_rate'])
		loss_fn = nn.CrossEntropyLoss()
		model, loss_fn = utils.move_to_cuda(config, model, loss_fn)
		return model, optimizer, loss_fn

def evaluate_slot(data, config, model, loss_fn,cut_ratio=-1, tie_ratio=-1,eval_t=False):	
	if (eval_t == True): #load validation set
		x_te, y_te, te_len, te_mask, text_te,map_te,subword_te, old_subword_mapping_te, tok_text_te, subword_te, label_tok_id_te, label_tok_mask_te, label_tok_name_te, label_subword_te, label_old_map_te, slot_te, focused_tie_label_te, full_tie_label_te = utils.load_data(data,'val')
	else:
		x_te, y_te, te_len, te_mask, text_te, map_te,subword_te, old_subword_mapping_te, tok_text_te, subword_te, label_tok_id_te, label_tok_mask_te, label_tok_name_te, label_subword_te, label_old_map_te, slot_te, focused_tie_label_te, full_tie_label_te = utils.load_data(data,'te')
 
	idx2slot={}
	for i in range (len(config['all_slots'])):
		idx2slot[i] = config['all_slots'][i]
	y_te_ind = utils.create_index(y_te)
	
	avg_loss = 0.0
	avg_acc = 0.0
	test_batch = int(math.ceil(x_te.shape[0] / config['batch_size'])) 

	cum_loss = 0.0
	
	model.eval()
	class_act, pred_act =[],[]
	correct_class_act, correct_pred_act=[],[]
	cum_test_reward=[]
	all_slots = config['all_slots']

	cl_example_info={'anchor':{}, 'pos':{}, 'neg':{}}

	if (config['model_type'] == 'cl'):
		dictionary = cl_utils.create_sampling_dict(y_te)

	temp_counter=0
	avg_cut, std_cut = [], []
	with torch.no_grad():
		for batch in range (test_batch):	
			begin_index = batch * (config['batch_size'])
			end_index = min((batch + 1) * config['batch_size'], x_te.shape[0])

			te_index = np.arange(begin_index,end_index)
			batch_pos={}
			batch_neg={}
			anchor={}
			return_anchor={}
			anchor = utils.extract_data(te_index, x_te, te_mask, text_te, y_te, slot_te, te_len, map_te, old_subword_mapping_te, tok_text_te, focused_tie_label_te, full_tie_label_te, subword_te, label_tok_id_te, label_tok_mask_te, label_tok_name_te, label_subword_te, label_old_map_te, anchor) 

			batch_focused_tie_label = anchor['focused_tie_label']
			batch_full_tie_label = anchor['full_tie_label']
		
			pos={}
			neg={}		
			new_idx =cl_utils. create_random_compare_idx(dictionary,anchor['y'],te_index,num_samples=1)	
			pos = utils.extract_data(new_idx, x_te, te_mask, text_te, y_te, slot_te, te_len, map_te, old_subword_mapping_te, tok_text_te, focused_tie_label_te, full_tie_label_te, subword_te,label_tok_id_te, label_tok_mask_te,label_tok_name_te, label_subword_te, label_old_map_te, pos) 

			neg_idx = cl_utils.create_random_neg_idx(dictionary, anchor['y'],num_samples=config['num_neg_samples'])	
			neg = utils.extract_data(neg_idx, x_te, te_mask, text_te, y_te, slot_te, te_len, map_te, old_subword_mapping_te, tok_text_te, focused_tie_label_te, full_tie_label_te, subword_te, label_tok_id_te, label_tok_mask_te,label_tok_name_te, label_subword_te, label_old_map_te, neg) 

			logits,label_logits, labels,_,_, return_anchor, return_pos,return_neg = model.forward(anchor, pos, neg)
				
			tie_break_pred= return_anchor['tie_break']
			
			pos_tie_break_pred = return_pos['tie_break']
		
			return_pos['text'] = pos['text']
			return_neg['text'] = neg['text']	
			
			return_pos['focused_tie_label'] = pos['focused_tie_label']	
			return_pos['full_tie_label'] = pos['full_tie_label']	

			return_neg['focused_tie_label'] = neg['focused_tie_label']	
			return_neg['full_tie_label'] = neg['full_tie_label']	
			#Extract incorrect spots for anchor	
			return_anchor['intent'] = anchor['y']

			return_anchor['focused_tie_label'] = anchor['focused_tie_label']	
			return_anchor['full_tie_label'] = anchor['full_tie_label']	
			return_anchor['text'] = anchor['text']	
			if (config['augment'] == 'true'):
				return_anchor['logits'] = logits.cpu().numpy()

			cl_example_info = pass_utils.update_np_info(cl_example_info, return_anchor, 'anchor')
			if (config['test_mode'] == 'true' and temp_counter == 1):
				print ("Break here")
				break

			temp_counter +=1
	all_slots = config['all_slots']

	full_metric, focus_metric = utils.obtain_tie_break_metric(cl_example_info['anchor']['focused_tie_label'], cl_example_info['anchor']['full_tie_label'], cl_example_info['anchor']['tie_break'])
	if (eval_t == True):
		focus_metric, full_metric = log_utils.write_result_report(focus_metric, full_metric, write='false') #Update F1 and compute H-Mean
	else:
		#log_utils.write_config(config)
		focus_metric, full_metric = log_utils.write_result_report(focus_metric, full_metric) #Update F1 and compute H-Mean
	
	#log_utils.print_len_info(all_ori_seg_len, all_pos_seg_len, all_neg_seg_len, all_ori_seg_len_std, all_pos_seg_len_std, all_neg_seg_len_std, config['model_type'])	
	return full_metric, focus_metric

	
def train_batch(data, config,clf_model,loss_fn, optimizer,pretrain=False):
        x_tr, y_tr, x_len_tr, tr_mask, tr_text,map_tr,subword_tr, old_subword_mapping_tr, tok_text_tr, subword_tr, label_tok_id_tr, label_tok_mask_tr, label_tok_name_tr, label_subword_tr, label_old_map_tr, slot_tr, focused_tie_break_tr, full_tie_break_tr = utils.load_data(data,'tr')

        temp_counter=0
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        all_classes = np.unique(y_tr)
        # Store original indexes of each class in dictionary
        idx_key = {}
        for val in all_classes:
                index = np.where(y_tr == val)[0]
                idx_key[val] = index

        all_slots = config['all_slots']

        idx2slot={}
        for i in range (len(config['all_slots'])):
                idx2slot[i] = config['all_slots'][i]
        best_acc = 0
        loss_acc = 0.0
        avg_acc = 0.0	
        avg_loss = 0.0
        full_avg={
                'break':{'p':0.0, 'r': 0.0,'f1': 0.0},	
                'tie':{'p':0.0,'r': 0.0,'f1': 0.0}		
        }

        focus_avg={
                'break':{'p':0.0,'r': 0.0,'f1': 0.0},	
                'tie':{'p':0.0,'r': 0.0,'f1': 0.0}		
        }

        type_label=['break', 'tie']
        type_metric=['p', 'r']

        early_stop_count = 0
        prev_loss =float("inf")

        num_batch = x_tr.shape[0] // config['batch_size']		
        print ("Num train batches ", num_batch, x_tr.shape[0])

        # Sanity check (Build dictionary of indexes for alll labels)
        unique_samples,unique_counts = np.unique(y_tr, return_counts=True)
        dictionary={}
        for label_id in unique_samples:
                indexes = np.argwhere(y_tr == label_id)
                dictionary[label_id] = indexes

        for idx in (dictionary.keys()):
                assert len(dictionary[idx]) == unique_counts[unique_samples == idx].item()
        start_time = time.time()
        for batch in range (num_batch):
                optimizer.zero_grad()
                anchor={}
                pos={}
                neg={}
                return_anchor={}

                #For training
                batch_index = train_utils.generate_batch(x_tr.shape[0], config['batch_size'])

                anchor = utils.extract_data(batch_index, x_tr, tr_mask, tr_text, y_tr, slot_tr, x_len_tr, map_tr, old_subword_mapping_tr, tok_text_tr, focused_tie_break_tr, full_tie_break_tr, subword_tr, label_tok_id_tr, label_tok_mask_tr, label_tok_name_tr, label_subword_tr, label_old_map_tr, anchor) 

                focused_batch_tie_break = anchor['focused_tie_label']
                full_batch_tie_break = anchor['full_tie_label']


                #Obtain a set of positive samples using intent label, then you have 2N samples => do similar thing as SimCLR
                pos_idx = cl_utils.create_random_compare_idx(dictionary,anchor['y'],batch_index, num_samples=1)	
                pos = utils.extract_data(pos_idx, x_tr, tr_mask, tr_text, y_tr, slot_tr, x_len_tr, map_tr, old_subword_mapping_tr, tok_text_tr, focused_tie_break_tr, full_tie_break_tr, subword_tr,label_tok_id_tr, label_tok_mask_tr,label_tok_name_tr, label_subword_tr, label_old_map_tr, pos) 
                        
                neg_idx = cl_utils.create_random_neg_idx(dictionary, anchor['y'],num_samples=config['num_neg_samples'])	
                neg = utils.extract_data(neg_idx, x_tr, tr_mask, tr_text, y_tr, slot_tr, x_len_tr, map_tr, old_subword_mapping_tr, tok_text_tr, focused_tie_break_tr, full_tie_break_tr, subword_tr,label_tok_id_tr, label_tok_mask_tr, label_tok_name_tr, label_subword_tr, label_old_map_tr, neg) 

                logits,label_logits, labels,self_score,self_labels, return_anchor,return_pos,return_neg = clf_model.forward(anchor, pos,neg)
                tie_break_pred = return_anchor['tie_break']

                loss_val= 0.0
                if (config['augment'] == 'true'):
                        rep_loss = loss_fn(logits, labels)
                        #print ("Sent Loss", rep_loss)
                else:
                        rep_loss = 0.0
                loss_val += config['sent_coeff'] * rep_loss
                if (config['self_supervise'] == 'true'):
                        self_loss = loss_fn(self_score, self_labels)
                        #print ("Seg Loss", self_loss)
                else:
                        self_loss = 0.0
                loss_val += config['seg_coeff'] * self_loss

                #print ("Loss val", loss_val)

                full_metric, focus_metric = utils.obtain_tie_break_metric(focused_batch_tie_break, full_batch_tie_break, tie_break_pred)

                torch.use_deterministic_algorithms(False)
                loss_val.backward()	
                torch.use_deterministic_algorithms(True)

                optimizer.step()
                avg_loss += loss_val.item()

                for t in type_label:
                        for m in type_metric:
                                focus_avg[t][m] += np.array(focus_metric[t][m]).mean() #focus: mean for one batch
                                full_avg[t][m] += np.array(full_metric[t][m]).mean()
                        focus_avg[t]['f1'] = 2*focus_avg[t]['p'] * focus_avg[t]['r'] / (focus_avg[t]['p'] + focus_avg[t]['r'] + 1e-20)
                        full_avg[t]['f1'] = 2*full_avg[t]['p'] * full_avg[t]['r'] / (full_avg[t]['p'] + full_avg[t]['r'] + 1e-20)

                focus_hmean = 2* focus_avg['tie']['f1'] * focus_avg['break']['f1'] / (focus_avg['tie']['f1'] + focus_avg['break']['f1'] + 1e-20)
                full_hmean = 2* full_avg['tie']['f1'] * full_avg['break']['f1'] / (full_avg['tie']['f1'] + full_avg['break']['f1'] + 1e-20)

                # Clean up CUDA Memory to free up space 
                if (torch.cuda.is_available()):
                        torch.cuda.empty_cache()

                if ((batch +1) % 50 ==0):

                        print ("---Batch {0:3d} \t Slot F1: {1:.2f} \t Loss: {2:.4f}----".format(batch+1, avg_acc/ (batch+1), avg_loss / (batch+1)))
                        print ("{0:10s} \t {1:5s} \t {2:5s} \t {3:5s} \t {4:5s} \t {5:5s} \t {6:5s} \t {7:5s}".format("Metric", "B-P", 'B-R', 'B-F1', 'T-P', 'T-R', 'T-F1', 'H-Mean'))
                        print ("{0:10s} \t {1:5.2f} \t {2:5.2f} \t {3:5.2f} \t {4:5.2f} \t {5:5.2f} \t {6:5.2f} \t {7:5.2f}".format("Focus", focus_avg['break']['p']*100/(batch+1), focus_avg['break']['r']*100/(batch+1), focus_avg['break']['f1']*100/(batch+1), focus_avg['tie']['p']*100/(batch+1), focus_avg['tie']['r']*100/(batch+1), focus_avg['tie']['f1']*100/(batch+1), focus_hmean * 100 / (batch+1)))
                        #print ("{0:10s} \t {1:5.2f} \t {2:5.2f} \t {3:5.2f} \t {4:5.2f} \t {5:5.2f} \t {6:5.2f} \t {7:5.2f}".format("Full",full_avg['break']['p']*100/(batch+1), full_avg['break']['r']*100/(batch+1), full_avg['break']['f1']*100/(batch+1), full_avg['tie']['p']*100/(batch+1), full_avg['tie']['r']*100/(batch+1), full_avg['tie']['f1']*100/(batch+1), full_hmean*100 / (batch+1)))
                        #break

                
                if (config['test_mode'] == 'true' and temp_counter == 1):
                        train_time = time.time() - start_time
                        print ("--Training time---", round(train_time,4))
                        break
                temp_counter +=1
                                
        return avg_acc, clf_model
					
def train_model(data,config, seed_val, model, optimizer, loss_fn):
	avg_acc = 0.0

	model.train()
	print ("---Training Starts---")
	best_eval_h_mean = 0.0
	early_stop = 5
	num_early_stop=0
	stop_epoch = 0
	for epoch in range(config['num_epochs']):
		avg_acc,model = train_batch(data, config, model, loss_fn, optimizer,pretrain=False)
		eval_full_metric, eval_focus_metric = evaluate_slot(data, config, model, loss_fn, eval_t=True) 
		eval_h_mean = eval_focus_metric['h_mean']
		stop_epoch += 1
		if (eval_h_mean > best_eval_h_mean):
			best_eval_h_mean = eval_h_mean
			print ("--- Saving model ---")
			torch.save(model.state_dict(), config['current_directory'] +'best_model.pth')
			print ("Best H-Mean now:", best_eval_h_mean)
		else:
			num_early_stop += 1	
		
		if (num_early_stop >= early_stop):
			print ("Early stop at epoch ", epoch)
			break	
		
		gc.collect()
		if (torch.cuda.is_available()):
			torch.cuda.empty_cache()

	print ("----Training Ends ------")
	gc.collect()
	if (torch.cuda.is_available()):
		torch.cuda.empty_cache()

	return stop_epoch


def load_pretrained_weights(best_model, current_directory):
	print ("--------Loading pretrained weights ----------")
	if (torch.cuda.is_available()):
		best_model.load_state_dict(torch.load(current_directory + 'best_model.pth'))
	else:
		best_model.load_state_dict(torch.load(current_directory + 'best_model.pth', map_location=torch.device('cpu')))
		print ("------- Done loading pretrained weights -------")
	return best_model

def conduct_exp(config,data, seed_val, log_writer):
	if (type(config['seg_level']) == int):
		config['seg_level'] = [config['seg_level']]
	
	model, optimizer, loss_fn = load_model(config, seed_val)
	early_stop = train_model(data,config, seed_val, model, optimizer, loss_fn) #multi-epoch training
			
	best_model = load_pretrained_weights(model, config['current_directory'])

	if (True):
		eval_full_metric, eval_focus_metric = evaluate_slot(data, config, model, loss_fn, eval_t=True) 

	test_full_metric, test_focus_metric = evaluate_slot(data, config, model, loss_fn, eval_t=False) 

	if (True):
		if (config['write_log'] == 'true'):
			log_writer.writerow([config['batch_size'], 
								config['self_supervise'],
								config['sent_temp'],
								config['seg_temp'],
								config['sent_coeff'],
								config['seg_coeff'],
								config['test_mode'],
								early_stop,
								config['num_epochs'],
								config['level_aug'], 
								config['mask_word_side'], 
								config['learning_rate'], 
								config['ratio_seg'], 
								eval_focus_metric['break']['f1'], 	
								eval_focus_metric['tie']['f1'], 	
								eval_focus_metric['h_mean'], 	

								test_focus_metric['break']['p'], 	
								test_focus_metric['break']['r'], 	
								test_focus_metric['break']['f1'], 	

								test_focus_metric['tie']['p'], 	
								test_focus_metric['tie']['r'], 	
								test_focus_metric['tie']['f1'], 	
								test_focus_metric['h_mean'], 	

						])
		report_h_mean = test_full_metric['h_mean']
	else:
		report_h_mean=0.0
	return report_h_mean

if __name__ == "__main__": 
        # load settings
        start_time = time.time()
        config = parse_argument()
        # load data
        print ("----------------- START MODEL --------------------------", flush=True)
        print ("------ Seg level:", config['seg_level'], "------", flush=True)
        if (torch.cuda.is_available()):
                print ("DEVICE IN USE", torch.cuda.current_device())
        # Training cycle
        if not (os.path.exists(config['ckpt_dir'])):
                utils.MakeDirectory(config['ckpt_dir'])

        num_run = config['num_run']


        results=[]
        train_utils.set_seed(config['cur_seed'])

        print ("Configuration", config)

        print ("============================================")
        print ("---------Experiment Run START ------- ")

        data = read_traditional.read_datasets(config['dataset'], [config['seg_level']],)
        all_slots = data['all_slots']
        config['all_slots'] = all_slots
        config['num_train_slot'] = len(all_slots)
        config['max_seg'] = data['max_seg']
        config['max_seg_len'] = data['max_len']
        config['num_intent'] = len(data['all_intents'])
        config['max_seg_len'] = data['max_len']

        if (config['write_log'] == 'true'):
                file_name = '../tune_log/' + config['dataset'] + '_tune_log.tsv'
                if (os.path.isfile(file_name)): #File exists
                        print ("File exists, stop write title")
                        write_title= True
                else:
                        write_title=True	
                log_file = open(file_name,'a')
                log_writer = csv.writer(log_file, delimiter='\t')	
                if (write_title == True):
                        log_writer.writerow(['bsz', 'self_supervise', 'soft_ratio', 'self_soft', 'loss_ratio', 'mask_loss_ratio','test_mode', 'early_stop_epoch', 'num_epoch', 'level_aug', 'surr_aug', 'lr', 'mask_ratio', 'eval_b_f1', 'eval_t_f1', 'eval_h', 'test_b_p', 'test_p_r', 'test_b_f1', 'test_t_p', 'test_t_r', 'test_t_f1', 'test_h'])
        else:
                print ("No log writer")
                log_writer=None
        current_directory = config['ckpt_dir'] + 'run_' + str(config['cur_seed']) + '/'

        if not (os.path.exists(current_directory)): # create ckpt if it does not exist yet
                utils.MakeDirectory(current_directory)	


        config['current_directory'] = current_directory
        test_acc = conduct_exp(config, data,config['cur_seed'],log_writer)
        results.append(test_acc)
        print ("---------- END MODEL --------------------------")
        print ("================================================")
        print ("----Elapsed Time:", time.time()-start_time)

