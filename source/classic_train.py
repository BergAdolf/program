"""
    train net
"""
import math
import torch
import os

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from source.dataset.classic_dataset import MyDataSet
from source.model.net import Net
from util.util import read_split_data, evaluate, train_one_epoch


def main():
    # extract data
    epochs = 10
    lr = 0.001
    lrf = 0.1
    batch_size = 32  # batch size
    save_path = "../data/weights/weight.pth"
    data_path = "../img/train/"
    nw = 1

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # x_train, y_train, x_valid, y_valid = data_unzip(DATA_PATH)
    if os.path.exists("../data/weights") is False:
        os.makedirs("../data/weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path, val_rate=0.8)

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label)

    train_ds = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           collate_fn=train_dataset.collate_fn)

    valid_ds = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           collate_fn=val_dataset.collate_fn)
    # loss_func = F.cross_entropy
    # loss_func = F.mse_loss
    model = Net(init_weights=True).to(dev)
    # loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # best_acc = 0
    # train_steps = len(train_ds)
    # valid_steps = len(valid_ds)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0
    best_pos = 0
    best_neg = 0

    for epoch in range(epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_ds,
                                    device=dev,
                                    epoch=epoch)
        scheduler.step()

        # validate
        acc, pos_num, neg_num = evaluate(model=model, data_loader=valid_ds, device=dev)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 5)))
        print("[epoch {}] positive accuracy: {}".format(epoch, pos_num.item()))
        print("[epoch {}] negative accuracy: {}".format(epoch, neg_num.item()))
        tb_writer = SummaryWriter(log_dir='../data/log/')
        tags = ["loss", "accuracy", "positive_accuarcy", "negative_accuarcy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], pos_num, epoch)
        tb_writer.add_scalar(tags[3], neg_num, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "../data/weights/model-{}.pth".format(epoch))

        if acc > best_acc:
            best_acc = acc
            best_pos = pos_num
            best_neg = neg_num
            torch.save(model.state_dict(), save_path)

    # save the best result
    print('best train result: %.3f' % best_acc)
    print('best train postive sample: %.3f' % best_pos.item())
    print('best train negative sample: %.3f' % best_neg.item())
    print('Finished Training')


if __name__ == '__main__':
    main()
