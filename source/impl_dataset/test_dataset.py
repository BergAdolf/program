"""
    extract data from train and test directory

"""

import torch

from source.dataset.classic_dataset import MyDataSet
from source.model.net import Net
from util import evaluate, read_valid_data


def main():
    batch_size = 30
    data_path = "../../img/valid"
    save_path = "../../data/weights/weight.pth"

    images_path, images_label = read_valid_data(data_path)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Net(init_weights=True).to(dev)
    model.load_state_dict(torch.load(save_path))
    dataset = MyDataSet(images_path=images_path, images_class=images_label)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              collate_fn=dataset.collate_fn)

    acc, pos_num, neg_num = evaluate(model=model, data_loader=data_loader, device=dev)
    print("[valid result] accuracy: {}".format(round(acc, 5)))
    print("[valid result] positive accuracy: {}".format(pos_num.item()))
    print("[valid result] negative accuracy: {}".format(neg_num.item()))


if __name__ == '__main__':
    main()
