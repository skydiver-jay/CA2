import os
from datetime import datetime
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications import ResNet50
from utils_tf2 import *

# is_plus = False
# if is_plus:
#     import CA2_SIM_TF2
# else:
#     import CA2_TF2

attack_method = "MIM_RO"
if attack_method == "CA2_SIM_TF2":
    import CA2_SIM_TF2
elif attack_method == "CA2_TF2":
    import CA2_TF2
elif attack_method == "MIM_RO":
    import MIM_RO


def ca2_tf2_demo():
    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = ["images/imagenet_06_609.jpg", "images/imagenet_01_559.jpg"]
    images = samples(bounds=bounds, batchsize=1, index=1, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    adv_x = CA2_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_ca2_basic_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:',
              tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])


def ca2_sim_tf2_demo():
    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = ["images/imagenet_06_609.jpg", "images/imagenet_01_559.jpg"]
    images = samples(bounds=bounds, batchsize=1, index=1, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    adv_x = CA2_SIM_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_ca2_sim_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:',
              tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])


def ca2_tf2(path):
    print("-- 初始化模型: ResNet50 --")
    model = ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = [path]
    images = samples(bounds=bounds, batchsize=1, index=0, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:', decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    adv_x = CA2_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_ca2_basic_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:', decode_predictions(model.predict(images))[0])


def ca2_sim_tf2(path):
    print("-- 初始化模型: ResNet50 --")
    model = ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = [path]
    images = samples(bounds=bounds, batchsize=1, index=0, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:', decode_predictions(model.predict(images))[0])

    print("-- 开始攻击 --")
    adv_x = CA2_SIM_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_ca2_sim_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:', decode_predictions(model.predict(images))[0])


def mim_ro(path):
    print("-- 初始化模型: ResNet50 --")
    model = ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = [path]
    images = samples(bounds=bounds, batchsize=1, index=0, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:', decode_predictions(model.predict(images))[0])

    print("-- 开始攻击 --")
    adv_x = MIM_RO.momentum_iterative_method(model_fn=model, x=tf.convert_to_tensor(images, tf.float32))

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_MIM_RO_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:', decode_predictions(model.predict(images))[0])


if __name__ == "__main__":

    image_paths = os.listdir("images")
    base_dir = "images/"

    count = 0
    total = len(image_paths)

    for image_path in image_paths:
        count += 1
        print("\n\n*********攻击开始( %d / %d)：%s*********" % (count, total, image_path))
        if attack_method == "CA2_SIM_TF2":
            print("*********使用PLUS版本攻击*********")
            ca2_sim_tf2(base_dir + image_path)
        elif attack_method == "CA2_TF2":
            print("*********使用BASIC版本攻击*********")
            ca2_tf2(base_dir + image_path)
        else:
            print("*********使用MIM RO版本攻击*********")
            mim_ro(base_dir + image_path)
        print("*********攻击结束( %d / %d)：%s*********\n\n" % (count, total, image_path))
