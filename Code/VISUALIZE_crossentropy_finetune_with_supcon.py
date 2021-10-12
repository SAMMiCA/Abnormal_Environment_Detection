import os
import pdb
import random
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score, silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import TensorDataset
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder


def remap_labels(pred_labels, true_labels):
    """Rename prediction labels (clustered output) to best match true labels."""
    # from itertools import permutations # import this into script.
    pred_labels, true_labels = np.array(pred_labels), np.array(true_labels)
    assert pred_labels.ndim == 1 == true_labels.ndim
    assert len(pred_labels) == len(true_labels)
    cluster_names = np.unique(pred_labels)
    accuracy = 0

    perms = np.array(list(permutations(np.unique(true_labels))))

    remapped_labels = true_labels
    for perm in perms:
        flipped_labels = np.zeros(len(true_labels))
        for label_index, label in enumerate(cluster_names):
            flipped_labels[pred_labels == label] = perm[label_index]

        testAcc = np.sum(flipped_labels == true_labels) / len(true_labels)
        if testAcc > accuracy:
            accuracy = testAcc
            remapped_labels = flipped_labels

    return accuracy, remapped_labels

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

test_dataset = ImageFolder(root='../Dataset/test/',transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)

n_classes = 3

resnet50 = models.resnet50(pretrained=False)

resnet50.fc = nn.Linear(2048,n_classes)

resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

resnet50.cuda()
# resnet50.load_state_dict(torch.load(f'./crossentropy_weight/minimum_loss.pth'))
resnet50.load_state_dict(torch.load(f'./crossentropy_weight/maximum_clustering_finetuning2.pth'))
device = 'cuda'

#feature extract
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}
# resnet50.fc.register_forward_hook(get_activation('fc'))
resnet50.eval()

result = []
label = []
with torch.no_grad():
    for k, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        label.append(y)
        output = resnet50(x).squeeze()
        result.append(output)
        _,pred = torch.max(output,1)

features = torch.cat(result)
features = features.cpu().numpy()
labels = torch.cat(label)
labels = labels.cpu().numpy()

cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'violet'}


#TSNE
tsne = TSNE(learning_rate=100, init='pca', random_state=1024)
transformed = tsne.fit_transform(features)
xs = transformed[:,0]
ys = transformed[:,1]

fig, ax = plt.subplots()

for g in np.unique(labels):
    ix = np.where(labels == g)
    ax.scatter(xs[ix], ys[ix], c = cdict[g], label = g)
ax.legend()

ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.title('Ground Truth')
plt.tight_layout()
plt.savefig('./pictures/after_GT')
plt.show()

#kmeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(features)

remapped_labels = remap_labels(kmeans.labels_,labels)[1].astype(int)

fig, ax2 = plt.subplots()
for g in np.unique(remapped_labels):
    # pdb.set_trace()
    ix = np.where(remapped_labels == g)
    ax2.scatter(xs[ix], ys[ix], c = cdict[g], label = g)

ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)

# plt.scatter(xs,ys,c = kmeans.labels_)
ax2.legend()
plt.title('K-Means Clustering')
plt.tight_layout()
plt.savefig('./pictures/after_KMeans')
plt.show()

print(jaccard_score(labels,remapped_labels,average=None))
print(f'Average : {np.mean(jaccard_score(labels,remapped_labels,average=None))}')
# print(jaccard_score(labels,dbscan.labels_,average=None))

print(f'NMI score : {metrics.normalized_mutual_info_score(labels,remapped_labels)}')


# remapped_labels  = Predicted labels
# labels           = True labels

# binart recall check
from sklearn.metrics import recall_score

# recall = recall_score(labels, remapped_labels)

def convert_binary(array, idx):
    arr = array.copy()
    indices = (arr==idx)
    arr[:]=0
    arr[indices] = 1
    return arr

binary_recall_abnormal = recall_score(convert_binary(labels,0),convert_binary(remapped_labels,0),pos_label=0) * 100


# clustering purity check
clustering_purity_average = 100*((labels == remapped_labels).sum())/len(remapped_labels)

clustering_purity_normal = 100*((labels[np.where(labels==0)]  == remapped_labels[np.where(labels==0)]).sum())/len(remapped_labels[np.where(labels==0)])
clustering_purity_fire = 100*((labels[np.where(labels==1)]  == remapped_labels[np.where(labels==1)]).sum())/len(remapped_labels[np.where(labels==1)])
clustering_purity_dark = 100*((labels[np.where(labels==2)]  == remapped_labels[np.where(labels==2)]).sum())/len(remapped_labels[np.where(labels==2)])
clustering_purity_dust = 100*((labels[np.where(labels==3)]  == remapped_labels[np.where(labels==3)]).sum())/len(remapped_labels[np.where(labels==3)])
pdb.set_trace()
