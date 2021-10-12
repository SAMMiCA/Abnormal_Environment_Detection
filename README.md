# Abnormal_Situation_Clustering

## Enviroment setup

To install requirements:
```
pip install -r requirements.txt
```



## Train

To train convolutional neural network(embedding):
```
cd Code
python TRAIN_crossentropy.py # for train unsupervised setting
python TRAIN_crossentropy_finetune_with_supcon.py # for finetuning
```




## Evaluation

To evaluate and visualize:
```
cd Code
python VISUALIZE_crossentropy.py # for train unsupervised setting
python VISUALIZE_crossentropy_finetune_with_supcon.py # for finetuning
```
