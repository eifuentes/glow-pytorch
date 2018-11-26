import random

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision.transforms as vtransforms
from glow.layers import *
from glow.model import *
from torchvision.datasets import LSUN

data_dir = '/Users/eifuentes/Data/lsun/church_outdoor'
img_rescale_size = 96
img_crop_size = 64
train_batch_size = 128
test_batch_sze = 8
num_workers = 16
num_levels = 4
num_flows_per_level = 48
num_input_channels = 3
affine = True
lu = False
lr = 0.001
beta1 = 0.9
beta2 = 0.999
max_epochs = 10
seed = None
cuda = False

seed = seed if seed else random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)

cuda = torch.cuda.is_available() and cuda
cudnn.benchmark = True

device = torch.device('cuda') if cuda else torch.device('cpu')

# some more preprocessing for quantization
train_data_transforms = vtransforms.Compose([
    vtransforms.Resize((img_rescale_size, img_rescale_size)),
    vtransforms.RandomCrop((img_crop_size, img_crop_size)),
    vtransforms.ToTensor(),
])
test_data_transforms = train_data_transforms

train_dataset = LSUN(root=data_dir, classes=['church_outdoor_train'], transform=train_data_transforms)
test_dataset = LSUN(root=data_dir, classes=['church_outdoor_val'], transform=test_data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_sze,
                             shuffle=False, num_workers=num_workers)

layer = Flow(3)
x, _ = next(iter(train_dataloader))
x.size()

y, logdet = layer(x)
y.size()
logdet

model = Glow(num_input_channels, num_levels, num_flows_per_level, affine, lu)
model.to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))


for epoch in range(1, max_epochs+1):
    # train loop
    model.train()
    for i, (x, _) in enumerate(train_dataloader, start=0):
        optimizer.zero_grad()
        z, logdet = model(x.to(device))
        loss = criterion()
        loss.backward()
        optimizer.step()
    # test loop
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(test_dataloader, start=0):
            z, logdet = model(x)
        # log some stats/plot some curves/save some images
    # save model
    # load from checkpoint, count in terms of num_iterations i.e epoch x batch size
