import os
import pdb
import warnings
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from pytorch_metric_learning import losses
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

warnings.simplefilter("ignore", UserWarning)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

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

def train(epoch, model, device, train_loader, optimizer, criterion_c):
    results = {
        'loss_c': 0
    }
    model.train()
    total_train = len(train_loader)
    result = []
    label = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)  # (batch_size, height, width)
        label.append(y)
        output = model(x).squeeze()
        result.append(output)
         # output = output.permute(0,2,3,1).contiguous().view(-1,5)
        # y = y.unsqueeze(1).permute(0,2,3,1).contiguous().view(-1)
        loss_c = criterion_c(output, y)
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        # if i % train_log_step == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss_C: {:.4f}'.format(epoch, 100, i, total_train, loss_c.item()))
        results['loss_c'] += loss_c.item()
    results['loss_c'] /= total_train

    return results

def validate(model, device, test_loader, criterion_c):
    results = {
        'loss_v': 0
    }
    model.eval()
    total_valid = len(test_loader)
    correct = 0.0
    result = []
    label = []
    with torch.no_grad():
        for k, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            label.append(y)

            model.avgpool.register_forward_hook(get_activation('avgpool'))

            output = model(x)

            loss_v = criterion_c(output, y)
            result.append(activation['avgpool'].squeeze())

            results['loss_v'] += loss_v.item()
    results['loss_v'] /= total_valid

    features = torch.cat(result)
    features = features.cpu().numpy()
    labels = torch.cat(label)
    labels = labels.cpu().numpy()

    cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'violet'}

    # TSNE
    tsne = TSNE(learning_rate=100, init='pca')
    transformed = tsne.fit_transform(features)
    xs = transformed[:, 0]
    ys = transformed[:, 1]

    kmeans = KMeans(n_clusters=3, random_state=0).fit(features)

    remapped_labels = remap_labels(kmeans.labels_, labels)[1]
    score = jaccard_score(labels, remapped_labels, average=None).mean()

    return results, score

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

train_dataset = ImageFolder(root='../Dataset/train_3/',transform=transform)
val_dataset = ImageFolder(root='../Dataset/val_3/',transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=84, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=84, shuffle=True, num_workers=1)

n_classes = 3

resnet50 = models.resnet50(pretrained=True)
# for param in resnet50.parameters():
#     param.requires_grad = False
resnet50.fc = nn.Linear(2048,n_classes)

backbone_params = [param for name, param in resnet50.named_parameters() if 'fc' not in name]
other_params = [param for name, param in resnet50.named_parameters() if 'fc' in name]

resnet50.cuda()
criterion = nn.CrossEntropyLoss()
device = 'cuda'

optimizer = optim.SGD([
    {"params":backbone_params, 'lr':3e-3},
    {"params":other_params, 'lr': 3e-2}],momentum=0.9)


loss_v = 10
score_c = 0.1

for epoch in range(100):
    print('Beginning Epoch {:02d}'.format(epoch))
    train_results = train(epoch, resnet50, device, train_loader, optimizer, criterion)
    print('TRAIN : LOSS_C: {:.4f} '.format(train_results['loss_c']))
    val_results, val_clustering_score = validate(resnet50, device, val_loader,  criterion)
    print('TEST : LOSS_V: {:.4f} '.format(val_results['loss_v']) + f'SCORE: {val_clustering_score}')
    if score_c < val_clustering_score:
        torch.save(resnet50.state_dict(), os.path.join('./crossentropy_weight/', 'maximum_clustering.pth'))
        score_c = val_clustering_score
        if score_c == 1.0:
            break
    if loss_v > val_results['loss_v']:
        torch.save(resnet50.state_dict(), os.path.join('./crossentropy_weight/', 'minimum_loss.pth'))
        loss_v = val_results['loss_v']
