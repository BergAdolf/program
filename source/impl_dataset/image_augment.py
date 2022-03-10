"""
    data augmentation
"""

from util.util import read_split_data


def main():
    data_path = "../../img/"
    file_dir = "../../data/carabas.pkl"
    zip_dir = "../../data/carabas.pkl.gz"

    x_train, y_train, x_valid, y_valid = read_split_data(data_path)


if __name__ == '__main__':
    main()
