import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data.sampler import BatchSampler
import pdb


dataset = ImageFolder(root='/home/jhjang/Desktop/AI28_3.1/Dataset')
labels = torch.ones(1360)
class Mydataset(Dataset):
    def __init__(self, image, label,Train):
        super(Dataset, self).__init__()
        self.Train = Train
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.anchor = image
        self.label = label

        if self.Train:
            self.labels_set = set(self.label.numpy())
            self.label_to_indices = {label: np.where(self.label.numpy() == label)[0]
                                     for label in self.labels_set}

    def __getitem__(self, index):
        if self.Train:
            anchor = self.anchor
            label = self.label

            positive =  self.flip(anchor)
            negative = 3
            return (anchor, positive, negative)
        else:
            return self.image[index], self.label[index]
    def __len__(self):
        return len(self.image)

d = Mydataset(dataset,labels,Train=True)
pdb.set_trace()

# train_dataset = Dataset(X_train_rnn, y_train)
train_dataset = Dataset(X_train_rnn, y_train,Train = True)
train_dataloader = DataLoader(train_dataset, batch_size=4000, shuffle=False)
test_dataset = Dataset(X_test_rnn, y_test,Train = False)
test_dataloader = DataLoader(test_dataset, batch_size=2000, shuffle=False)


class TripletELECTRICTUNNEL(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.train_labels
            self.train_data = self.dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)