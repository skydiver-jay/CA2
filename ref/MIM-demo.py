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
    print("\n---- 样本的top标签 %d : %s ----" % (np.argmax(images_label), filename))
    print('样本的top分类: ', decode_predictions(model.predict(images), top=2)[0])


if __name__ == "__main__":
    print("tensorflow's version is: ")
    print(tf.__version__)
    print("\n")

    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 使用Ditto.use_define_samples读取攻击目标图像 --")
    paths = ["imagenet_06_609.jpg", "imagenet_01_559.jpg"]
    images, labels = use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=bounds,
                                                batchsize=1, index=1, paths=paths, user_define_labels=[243])

    original_label = model(images)
    print("使用LocalModel预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用LocalModel预测的目标图像的top分类:', decode_predictions(model.predict(images), top=2)[0])

    # images_target, labels_target = use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=0, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    adv_x = momentum_iterative_method.momentum_iterative_method(model_fn=model, x=images, eps=3.0, eps_iter=3.0 / 16.0,
                                                                nb_iter=16)
    # adv_x = momentum_iterative_method.momentum_iterative_method(model_fn=model, x=images, eps=6.0, eps_iter=6.0 / 16.0,
    #                                                             nb_iter=16, y=tf.argmax(model(images_target), 1),
    #                                                             targeted=True)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用LocalModel预测的adv图像的top标签: ", np.argmax(adv_x_label))





    # exit(1)

    if utils_ditto.is_adv(original_label, adv_x_label):
        filename = "output/adv_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print("\n---- 攻击得出的对抗样本的top标签 %d : %s ----" % (int(np.argmax(adv_x_label)), filename))
        utils_ditto.save_image(adv_x[0], filename)
        images, labels = use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=bounds,
                                                    batchsize=1, index=0, paths=[filename])
        adv_x_label_save = model(images)
        print("使用LocalModel预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用LocalModel预测的保存的adv图像的top分类:', decode_predictions(model.predict(images), top=2)[0])

        # sample_and_show(model, filename)

        # print("-- 展示攻击得出的对抗样本 --")
        # utils_ditto.show_image('', model, adv_x[0], 'Adversarial image')



