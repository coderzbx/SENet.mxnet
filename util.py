# -*-coding:utf-8-*-

import os
import numpy as np
from functools import partial
from PIL import Image
from multiprocessing import Pool
import mxnet as mx


cfg = {}


def load_weights(weights_file):
    assert os.path.exists(weights_file)
    name_list = weights_file.split("-")
    prefix = "-".join(name_list[:-1])
    epoch = int(name_list[-1].split(".")[0])
    print ("prefix: {}, epoch: {}".format(prefix, epoch))
    network, net_args, net_auxs = mx.model.load_checkpoint(prefix, epoch)
    return network, net_args, net_auxs


def color_scale(scale):
    """ColorScale.
    """

    def _impl(data):
        data = data.astype(np.float32, copy=False)
        data *= np.float32(scale)
        return data

    return _impl


def color_normalize(mean, std=None):
    """ColorNormalize.
    """

    def _impl(data):
        data = data.astype(np.float32, copy=False)
        data -= np.array(mean, dtype=np.float32)
        if std is not None:
            data /= np.array(std, dtype=np.float32)
        return data

    return _impl


def _interp_preds_as_impl(imh, imw, pred):
    """
        interpolate each dimension
    """
    pred = pred.astype(np.single, copy=False)
    input_h, input_w = pred.shape[:2]
    if input_h == imh:
        interp_pred = pred
    else:
        # interp_method = Image.BILINEAR
        interp_method = get_interp_method(input_h, input_w, imw, imh)
        interp_pred = np.array(Image.fromarray(pred).resize((imw, imh), interp_method))
    return interp_pred


def interp_preds_as(net_preds, im_shape, threads=4):
    """
        net_preds : (C, H, W)
        im_shape  : (H, W, C)
    """
    imh, imw = im_shape[:2]
    worker = partial(_interp_preds_as_impl, imh, imw)
    if threads == 1:
        ret = [worker(_) for _ in net_preds]
    else:
        pool = Pool(threads)
        ret = pool.map(worker, net_preds)
        pool.close()
        pool.join()
    return np.array(ret)


def get_interp_method(imh_src, imw_src, imh_dst, imw_dst, default=3):
    """
        get_interp_method
    """
    if not cfg.get('choose_interpolation_method', False):
        return default
    if imh_dst < imh_src and imw_dst < imw_src:
        return Image.ANTIALIAS
    elif imh_dst > imh_src and imw_dst > imw_src:
        return Image.CUBIC
    else:
        return Image.LINEAR