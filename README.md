# Slot Induction via Pre-trained Language Model Probing and Multi-level Contrastive Learning
This repository provides PyTorch implementation for the paper [*Slot Induction via Pre-trained Language Model Probing and Multi-level Contrastive Learning*](https://arxiv.org/pdf/2308.04712.pdf) **(SIGDIAL 2023)**


## Environment Setup
The simplest way to set up the environment is to run our prepared BASH script as follows (**NOTE:** Anaconda virtual environment needs to be pre-installed before the scripts)

```
bash setup_env.sh
```

**Your own virtual environment**: Make sure you have python==3.9.12 and PyTorch==1.12.1 properly installed. Then, simply use pip to install the remaining required packages:

```
pip install -r requirements.txt
```


## Dataset
We provide our prepared split of P1 and P2 datasets for both SNIPS and ATIS datasets under './dataset/' (See our paper for further details). 


## Configuration
Major important arguments are (configurable within the corresponding *run_model.sh* script):

* ```--ckpt_dir```: Saved directory for checkpoint
* ```--epoch```: Number of training epochs
* ```--lr```: training learning rate
* ```--dataset```: Choose dataset to train/evaluate (i.e. SNIPS_P1/ ATIS_P1)

**Tuning hyperparameters**

* ```seg_level```: Depth level of segmentation tree to extract semantic segments ($d$)
* ```sent_temp```: SentCL temperature ($\tau_{d}$)
* ```seg_temp```: SegCL temperature ($\tau_{s}$)

* ```sent_coeff```: Coefficient for SentCL loss ($\gamma$)
* ```seg_coeff```: Coefficient for SegCL loss ($\delta$)

**Optional Configuration**
* ```ratio_seg```: Ratio of segments for cropping (augmentations)
* ```mask_type```: Options to apply augmentations or not (i.e. no_mask or mask_seg)



## Running Experiments

The following scripts are for Slot Induction training and evaluation (P1). </br>
**SNIPS**
```
cd ./code/script/trad/
bash run_model.sh
```

**SNIPS**
```
cd ./code/script/atis/
bash run_model.sh
```

## Citation
If you find our ideas, code or dataset helpful, please consider citing our work as follows:
<pre>
@article{nguyen2023slot,
  title={Slot Induction via Pre-trained Language Model Probing and Multi-level Contrastive Learning},
  author={Nguyen, Hoang H and Zhang, Chenwei and Liu, Ye and Yu, Philip S},
  journal={arXiv preprint arXiv:2308.04712},
  year={2023}
}
</pre>

## Acknowledgement
Our UPL implementation is adapted from [Petrurbed Masking](https://github.com/LividWo/Perturbed-Masking)  </br>
Our dataset is adapted from [Capsule-NLU](https://github.com/czhang99/Capsule-NLU) </br>
