import os
import glob
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from model_v2 import MobileNetV2
import matplotlib.pyplot as plt

img_data_dir = "face_data_1"
save_weights_path = "./save_weights/resMobileNetV2.ckpt"


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root, "data_set", img_data_dir)
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    log_save_path = os.path.join(data_root, "loss log")
    assert os.path.exists(log_save_path)

    loss_path = "loss_1.jpg"
    accuracy_path = "acc_1.jpg"
    plt_save_path_loss = os.path.join(log_save_path, loss_path)
    plt_save_path_accuracy = os.path.join(log_save_path, accuracy_path)

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    num_classes = 2

    def pre_function(img):
        # img = im.open('test.jpg')
        # img = np.array(img).astype(np.float32)
        img = img / 255.
        img = (img - 0.5) * 2.0
        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                               preprocessing_function=pre_function)

    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    # img, _ = next(train_data_gen)
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    # create model except fc layer
    feature = MobileNetV2(num_classes=num_classes, include_top=False)
    # download weights 链接: https://pan.baidu.com/s/1YgFoIKHqooMrTQg_IqI2hA  密码: 2qht
    pre_weights_path = './tf_mobilenet_weights/pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path+"*")), "cannot find {}".format(pre_weights_path)
    feature.load_weights(pre_weights_path)
    feature.trainable = False
    feature.summary()

    # add last fc layer
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
    model.summary()

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def val_step(images, labels):
        output = model(images, training=False)
        loss = loss_object(labels, output)

        val_loss(loss)
        val_accuracy(labels, output)

    x_list = []
    train_loss_y_list = []
    train_accuracy_y_list = []
    val_loss_y_list = []
    val_accuracy_y_list = []

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(range(total_train // batch_size))
        for step in train_bar:
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # for循环中epoch从0开始
        x_list.append(epoch+1)
        train_loss_y_list.append(train_loss.result().numpy())
        train_accuracy_y_list.append(train_accuracy.result().numpy())

        # validate
        val_bar = tqdm(range(total_val // batch_size))
        for step in val_bar:
            val_images, val_labels = next(val_data_gen)
            val_step(val_images, val_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())

        val_loss_y_list.append(val_loss.result().numpy())
        val_accuracy_y_list.append(val_accuracy.result().numpy())

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights(save_weights_path, save_format="tf")

    plt.figure('loss log')
    ax = plt.gca()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(x_list, train_loss_y_list, color='red', linewidth=1, alpha=0.6)
    ax.plot(x_list, val_loss_y_list, color='purple', linewidth=1, alpha=0.6)
    plt.legend(handles=[11, 12], labels=['train', 'val'], loc='best')
    plt.savefig(plt_save_path_loss)

    plt.figure('accuracy log')
    ax = plt.gca()
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.plot(x_list, train_accuracy_y_list, color='blue', linewidth=1, alpha=0.6)
    ax.plot(x_list, val_accuracy_y_list, color='cyan', linewidth=1, alpha=0.6)
    plt.legend(handles=[11, 12], labels=['train', 'val'], loc='best')
    plt.savefig(plt_save_path_accuracy)

    plt.show()


if __name__ == '__main__':
    main()
