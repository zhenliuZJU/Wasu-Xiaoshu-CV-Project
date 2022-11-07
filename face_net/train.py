from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

data_dir = './data/dorm'

batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8
SAVE_PATH = f'./result_epochs_{epochs}.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
    for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# Remove mtcnn to reduce GPU memory usage
del mtcnn

resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(dataset.class_to_idx)
).to(device)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training_correct = 0
    training_total = 0
    for i, (x, y) in enumerate(train_loader):
        output = resnet(x.to(device))
        loss = loss_fn(output, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = torch.argmax(output.data, 1)
        training_correct += (y_pred == y.to(device)).sum().item()
        training_total += y.size(0)
        accuracy = training_correct / training_total
        print(f"-------> training accuracy: {accuracy} || loss: {loss.data}")

    scheduler.step()

    resnet.eval()
    val_correct = 0
    val_total = 0
    for i, (x, y) in enumerate(val_loader):
        output = resnet(x.to(device))
        y_pred = torch.argmax(output.data, 1)
        val_correct += (y_pred == y.to(device)).sum().item()
        val_total += y.size(0)
        accuracy = val_correct / val_total
        print(f"-------> validation accuracy: {accuracy}")

ckpt = {
    'epoch':epoch,
    'classes':dataset.classes,
    'model':resnet,
}
torch.save(ckpt, SAVE_PATH)
