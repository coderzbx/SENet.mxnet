# -*-coding:utf-8-*-

import os
import argparse
import time
import logging
import shutil


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 3:
                print('lst should at least has three parts, but only has %s parts for %s' %(line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' %(line, e))
                continue
            yield item


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--list', help='prefix of input/output lst and rec files.', required=True)
    parser.add_argument('--dir', help='directory to save image', required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logger = logging.getLogger()
    args = parse_args()

    start_time = time.time()

    count = 0

    lst_file = args.list

    image_list = read_list(lst_file)
    image_list = list(image_list)
    count = len(image_list)

    check_dir = args.dir

    if str(lst_file).endswith("train.lst"):
        check_dir = os.path.join(check_dir, "train")
    elif str(lst_file).endswith("val.lst"):
        check_dir = os.path.join(check_dir, "val")
    elif str(lst_file).endswith("test.lst"):
        check_dir = os.path.join(check_dir, "test")

    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
        
    class_dict = {}
    for label_info in image_list:
        label_path = label_info[1]
        label_id = int(label_info[2])

        if label_id not in class_dict:
            class_dict[label_id] = []
        class_dict[label_id].append(label_path)

        class_dir = os.path.join(check_dir, str(label_id))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        image_path = os.path.join(class_dir, os.path.basename(label_path))
        shutil.copy(label_path, image_path)

    for label_id, labels in class_dict.items():
        print ("{}: {}/{}".format(str(label_id), str(len(labels)), count))

    end_time = time.time()
    print ("finish in {} s".format(end_time - start_time))

