import foolbox as fb
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import io, os

import use_define_samples, utils_ditto, momentum_iterative_method


def sample_and_show(model, path):

    images, labels = use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=(0, 255),
                                                batchsize=1, index=0, paths=[path])
    images_label = model(images)
    print("\n---- 样本的top标签 %d : %s ----" % (np.argmax(images_label), path))
    print('---- 样本的top 2 分类: %s ----:' % str(decode_predictions(model.predict(images), top=2)[0]))
    print(utils_ditto.b64_encode_image(path))


if __name__ == "__main__":
    print("tensorflow's version is: ")
    print(tf.__version__)
    print("\n")

    print("Get model: ResNet50 ----")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    sample_and_show(model, "imagenet_01_559.jpg")
    sample_and_show(model, "output/adv_20230601-103813.jpg")
    sample_and_show(model, "../TF2_dev/output/adv_ca2_basic_20230605-174521.jpg")
    sample_and_show(model, "../TF2_dev/output/adv_ca2_sim_20230605-171639.jpg")
