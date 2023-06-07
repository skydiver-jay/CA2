import foolbox as fb
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications import ResNet50
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import io, os

import utils_tf2


def sample_and_show(model, path):
    images = utils_tf2.samples(bounds=(0, 255), batchsize=1, index=0, paths=[path], shape=(224, 224))
    images_label = model(images)
    print("\n---- 样本的top标签 %d: %s ----" % (int(np.argmax(images_label)), path))
    print('---- 样本的top 2 分类: %s ----:' % str(decode_predictions(model.predict(images), top=2)[0]))
    print(utils_tf2.b64_encode_image(path))


if __name__ == "__main__":
    print("TensorFlow版本: %s\n" % tf.__version__)

    print("---- 初始化模型: ResNet50 ----")
    model = ResNet50(weights="imagenet")
    bounds = (0, 255)

    base_dir = "output/"
    adv_paths = os.listdir(base_dir)

    ca2_sim_count = 0
    ca2_basic_count = 0
    mim_ro_count = 0

    # for adv_path in adv_paths:
    #     if "20230607" in adv_path:
    #         if "ca2_sim" in adv_path:
    #             ca2_sim_count += 1
    #         elif "ca2_basic" in adv_path:
    #             ca2_basic_count += 1
    #         elif "MIM_RO" in adv_path:
    #             mim_ro_count += 1
    #         sample_and_show(model, base_dir + adv_path)
    #
    # print("ca2_sim_count: ", ca2_sim_count)
    # print("ca2_basic_count: ", ca2_basic_count)
    # print("mim_ro_count: ", mim_ro_count)

    # base_dir = "images/"
    # image_paths = os.listdir(base_dir)
    #
    # for image_path in image_paths:
    #     sample_and_show(model, base_dir + image_path)
    #
    # print("original_img_count: ", len(image_paths))

    sample_and_show(model, "output/adv_ca2_basic_13_471_alone_20230607-235330.jpg")

