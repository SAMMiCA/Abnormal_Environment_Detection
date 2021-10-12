# Abnormal_Situation_Clustering

## Enviroment setup

To install requirements:
```
pip install -r requirements.txt
```


## Dataset

Please download dataset here:
- [Dataset](https://kaistackr-my.sharepoint.com/:u:/g/personal/jhyuk_kaist_ac_kr/EcaeOeoYRGZKud2pInUuDU0BFlmcYNhiHwzSX6rJTXyyPA?e=reWSl6):


## Pre-trained Weights

Please download weights here
- [ResNet50 Weights](https://kaistackr-my.sharepoint.com/:u:/g/personal/jhyuk_kaist_ac_kr/ES-xEZwXYN9CnPrx_9ZN_P4BEznkZlGD10iwPVrz3vo1DQ?e=oyUj4k):

## Train

To train convolutional neural network (embedding):
```
cd Code
python TRAIN_crossentropy.py # for training unsupervised setting
python TRAIN_crossentropy_finetune_with_supcon.py # for finetuning
```


## Evaluation

To evaluate and visualize before finetuning:
```
cd Code
python VISUALIZE_crossentropy.py
```
<div align="center">
  <img width="30%" alt="1" src="./Code/pictures/before_GT.png">
  <img width="30%" alt="1" src="./Code/pictures/before_KMeans.png">
</div>
<div align="center">
  Clustering Results before Finetuning.
</div>

To evaluate and visualize after finetuning:
```
cd Code
python VISUALIZE_crossentropy_finetune_with_supcon.py
```
<div align="center">
  <img width="30%" alt="1" src="./Code/pictures/after_GT.png">
  <img width="30%" alt="1" src="./Code/pictures/after_KMeans.png">
</div>
<div align="center">
  Clustering Results after Finetuning.
</div>

