# -*-coding:utf-8-*-

import sys
sys.path.insert(0, "/opt/densenet.mxnet")

import numpy as np
import os
import json
import mxnet as mx
import time
import argparse
import cv2
from PIL import Image
from collections import namedtuple
import shutil

from util import load_weights
from label_file import main_assist_labels

# define a simple data batch
Batch = namedtuple('Batch', ['data'])


class ModelClassfication:
    def __init__(self, gpu_id=0):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)

        # model_file_name = "densenet-kd-169-0-5000.params"
        self.weights = model_file

        network, net_args, net_auxs = load_weights(self.weights)
        context = [mx.gpu(gpu_id) if gpu_id >= 0 else mx.cpu()]
        self.mod = mx.mod.Module(network, context=context)

        self.input_shape = [256, 256] # (W, H)
        # self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.input_shape[1], self.input_shape[0]))],
        #               label_shapes=[('softmax_label', (1,))])
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.input_shape[1], self.input_shape[0]))],
                      label_shapes=None)
        self.mod.init_params(arg_params=net_args,
                        aux_params=net_auxs)
        self._flipping = False

    def do(self, image_data):
        pred_data = None
        accuracy = 0
        try:
            print("start to load image.........")
            time1 = time.time()
            image = np.asarray(bytearray(image_data), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            print ("original img size:{}".format(img.shape))
            time2 = time.time()
            print("finish to load image, use {} s".format(time2-time1))

            # pad image
            # img = np.array(Image.fromarray(origin_frame.astype(np.uint8, copy=False)))
            print("start to resize image.........")
            newsize = max(img.shape[:2])
            new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
            margin0 = (newsize - img.shape[0]) // 2
            margin1 = (newsize - img.shape[1]) // 2
            new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
            # img: (256, 256, 3), GBR format, HWC
            img = cv2.resize(new_img, tuple(self.input_shape))
            print ("resized img size:{}".format(img.shape))
            time3 = time.time()
            print("finish to resize image, use {} s".format(time3 - time2))

            print("start to predict.........")
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :]

            # compute the predict probabilities
            self.mod.forward(Batch([mx.nd.array(img)]))
            prob = self.mod.get_outputs()[0].asnumpy()

            # Return the top-5
            prob = np.squeeze(prob)
            acc = np.sort(prob)[::-1]
            a = np.argsort(prob)[::-1]

            pred_data = a[0:5]
            accuracy = acc[0:5]
            time4 = time.time()
            print("finish to preditc, use {} s".format(time4 - time3))

        except Exception as e:
            print("recognition error:{}".format(repr(e)))

        return pred_data, accuracy

if __name__ == "__main__":
    time_start = time.time()

    name_dict = {}
    label_dict = {}
    for label in main_assist_labels:
        if label.categoryId not in name_dict:
            name_dict[label.categoryId] = label.name
            name_dict[label.label] = label.name
        if label.categoryId not in label_dict:
            label_dict[label.categoryId] = label.label

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0, required=False)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        print("model file[{}] is not exist".format(model_file))
        exit(0)

    image_dir = args.dir
    dest_dir = args.dest_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    model_net = ModelClassfication(gpu_id=args.gpu)
    proc_list = []

    print("loading test label...\n")
    label_map = {}
    total_val = 0
    recall_map = {}

    dir_list = os.listdir(image_dir)
    for id_dir in dir_list:
        class_dir = os.path.join(image_dir, id_dir)
        file_list = os.listdir(class_dir)
        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue

            proc_list.append(os.path.join(image_dir, id_dir, file_id))
            class_id = id_dir
            label_map[file_id] = class_id
            total_val += 1

            if class_id not in recall_map:
                recall_map[class_id] = {"total": 1}
            else:
                recall_map[class_id]["total"] += 1

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for id_ in proc_list:
        file_path = id_

        try:
            pred_label = None
            start = time.time()
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                img = f.read()
                pred_label, accuracy = model_net.do(image_data=img)
            end = time.time()

            class_id = pred_label[0]
            class_id = label_dict[int(class_id)]
            class_acc = accuracy[0]
            class_dir = os.path.join(dest_dir, str(class_id))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            dest_path = os.path.join(class_dir, os.path.basename(id_))
            shutil.copy(file_path, dest_path)
            print("Processed {} in {} ms,acc:{}, labels:{} vs. {}".format(
                os.path.basename(dest_path), str((end - start) * 1000),
                class_acc, str(class_id), label_map[os.path.basename(file_path)])
            )
        except Exception as e:
            print (repr(e))

    time_end = time.time()
    print("finish recognition in {} s\n".format(time_end-time_start))

    print("start to calculate recall and accuracy...")
    print("loading prediction...")
    accuracy_map = {}

    pred_map = {}
    class_dir_list = os.listdir(dest_dir)
    for class_dir in class_dir_list:
        pred_dir = os.path.join(dest_dir, class_dir)
        pred_list = os.listdir(pred_dir)

        for pred_file in pred_list:
            if not pred_file.endswith("jpg"):
                continue
            pred_map[pred_file] = class_dir

            if class_dir not in accuracy_map:
                accuracy_map[class_dir] = {"total": 1}
            else:
                accuracy_map[class_dir]["total"] += 1

    correct_map = {}

    for image_name, class_id in label_map.items():
        pred_class = pred_map[image_name]

        if class_id == pred_class:
            if class_id not in correct_map:
                correct_map[class_id] = 1
            else:
                correct_map[class_id] += 1

    print("start to calculate recall and accuracy...")

    for class_id, count in correct_map.items():
        recall_map[class_id]["correct"] = count
        recall_map[class_id]["rate"] = float(count) / float(recall_map[class_id]["total"]) * 100

        accuracy_map[class_id]["correct"] = count
        accuracy_map[class_id]["rate"] = float(count) / float(accuracy_map[class_id]["total"]) * 100

    # for class_id, info in recall_map.items():
    #     label_name = name_dict[class_id]
    #     print(label_name.encode("UTF-8"))
    #     print("recall:id:{},info:{}".format(class_id, json.dumps(info)))

    # for class_id, info in accuracy_map.items():
    #     label_name = name_dict[class_id]
    #     print(label_name.encode("UTF-8"))
    #     print("accuracy:id:{},info:{}".format(class_id, json.dumps(info)))

    # format
    print("recall, id, rate, correct/total")
    for class_id, info in recall_map.items():
        rate = 0
        if 'rate' in info:
            rate = info['rate']
        correct = 0
        if 'correct' in info:
            correct = info['correct']
        total = 0
        if 'total' in info:
            total = info['total']

        label_name = name_dict[class_id]
        print("{},{},{},{}/{}".format(label_name.encode("UTF-8"), class_id, rate, correct, total))

    print("\naccuracy, id, rate, correct/total")
    for class_id, info in accuracy_map.items():
        rate = 0
        if 'rate' in info:
            rate = info['rate']
        correct = 0
        if 'correct' in info:
            correct = info['correct']
        total = 0
        if 'total' in info:
            total = info['total']

        label_name = name_dict[class_id]
        print("{},{},{},{}/{}".format(label_name.encode("UTF-8"), class_id, rate, correct, total))
