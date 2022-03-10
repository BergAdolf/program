"""
    train net
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from source.dataset.dataset import MyDataSet
from source.model.net import Net
from util import data_unzip

# extract data
epochs = 200
lr = 0.0001
bs = 32  # batch size
save_path = "../data/weights/weight.pth"
DATA_PATH = '../data/carabas.pkl.gz'

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_train, y_train, x_valid, y_valid = data_unzip(DATA_PATH)
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
# map to torch format
x_train = x_train.type(torch.FloatTensor)
y_train = y_train.type(torch.FloatTensor)
x_valid = x_valid.type(torch.FloatTensor)
y_valid = y_valid.type(torch.FloatTensor)

# x_train 就是训练图片，y_train 就是训练标签
# x_valid 就是验证图片，y_valid 就是验证标签
train_ds = MyDataSet(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = MyDataSet(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True)

# define loss_func
# loss_func = F.cross_entropy
loss_func = F.mse_loss

model = Net(init_weights=True).to(dev)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
best_acc = 0
train_steps = len(train_ds)
valid_steps = len(valid_ds)

for epoch in range(epochs):
    # train
    model.train()
    train_accurate = 0.0
    val_accurate = 0.0
    running_loss = 0.0

    for images, labels in train_dl:
        outputs = model(images.to(dev))
        loss = loss_function(outputs, labels.to(dev))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    # validate
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for images, labels in train_dl:
            outputs = model(images.to(dev))
            outputs[outputs < 0.5] = 0
            outputs[outputs > 0.5] = 1
            train_accurate += torch.eq(outputs, labels.to(dev)).sum().item()

    train_accurate = train_accurate / train_steps
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss, train_accurate))

    with torch.no_grad():
        for images, labels in valid_dl:
            outputs = model(images.to(dev))
            outputs[outputs < 0.5] = 0
            outputs[outputs > 0.5] = 1
            val_accurate += torch.eq(outputs, labels.to(dev)).sum().item()

    val_accurate = val_accurate / valid_steps
    print('[epoch %d] val_accuracy: %.3f' % (epoch + 1, val_accurate))
    print('--------------------******------------------------')
    # save the best result
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(model.state_dict(), save_path)

print('best train result: %.3f' % best_acc)
print('Finished Training')
