import foolbox as fb
import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import io, os

import ref.use_define_samples, ref.utils_ditto

is_plus = True
if is_plus:
    import CA2_SIM_TF2
else:
    import CA2_TF2


def ca2_tf2_demo():
    # print("tensorflow's version is: %s" % tf.__version__)
    # print("\n")

    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 使用Ditto.use_define_samples读取攻击目标图像 --")
    paths = ["../ref/imagenet_06_609.jpg", "../ref/imagenet_01_559.jpg"]
    images, labels = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=bounds,
                                                    batchsize=1, index=1, paths=paths)

    original_label = model(images)
    print("使用LocalModel预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用LocalModel预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    # adv_x = MIM_RO_V3.momentum_iterative_method(model_fn=model, x=images,
    #                                             y=tf.argmax(model(images_target), 1), targeted=True)
    adv_x = CA2_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用LocalModel预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    # exit(1)

    if ref.utils_ditto.is_adv(original_label, adv_x_label):
        filename = "output/adv_ca2_basic_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        ref.utils_ditto.save_image(adv_x[0], filename)
        images, labels = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=bounds,
                                                        batchsize=1, index=0, paths=[filename])
        adv_x_label_save = model(images)
        print("使用LocalModel预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用LocalModel预测的保存的adv图像的top分类:',
              tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

        # sample_and_show(model, filename)

        # print("-- 展示攻击得出的对抗样本 --")
        # ref.utils_ditto.show_image('', model, adv_x[0], 'Adversarial image')


def ca2_sim_tf2_demo():
    # print("tensorflow's version is: %s" % tf.__version__)
    # print("\n")

    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 使用Ditto.use_define_samples读取攻击目标图像 --")
    paths = ["../ref/imagenet_06_609.jpg", "../ref/imagenet_01_559.jpg"]
    images, labels = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=bounds,
                                                    batchsize=1, index=1, paths=paths)

    original_label = model(images)
    print("使用LocalModel预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用LocalModel预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    # adv_x = MIM_RO_V3.momentum_iterative_method(model_fn=model, x=images,
    #                                             y=tf.argmax(model(images_target), 1), targeted=True)
    adv_x = CA2_SIM_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用LocalModel预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    # exit(1)

    if ref.utils_ditto.is_adv(original_label, adv_x_label):
        filename = "output/adv_ca2_sim_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        ref.utils_ditto.save_image(adv_x[0], filename)
        images, labels = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet', bounds=bounds,
                                                        batchsize=1, index=0, paths=[filename])
        adv_x_label_save = model(images)
        print("使用LocalModel预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用LocalModel预测的保存的adv图像的top分类:',
              tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

        # sample_and_show(model, filename)

        # print("-- 展示攻击得出的对抗样本 --")
        # ref.utils_ditto.show_image('', model, adv_x[0], 'Adversarial image')


if __name__ == "__main__":

    if is_plus:
        print("*********测试PLUS版本*********")
        ca2_sim_tf2_demo()
    else:
        print("*********测试BASIC版本*********")
        ca2_tf2_demo()

