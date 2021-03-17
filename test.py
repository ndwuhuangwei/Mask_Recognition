from predict import load_model, predict_img
import os
from PIL import Image
import numpy as np
import json
import glob
import tensorflow as tf
from predict import load_model

my_model = load_model()

my_img_dir = "./test_data"
im_height = 224
im_width = 224
num_classes = 2

for test_img in os.listdir(my_img_dir):
    print(test_img)
    img_path = os.path.join(my_img_dir, test_img)

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # resize image to 224x224
    img = img.resize((im_width, im_height))
    # plt.imshow(img)

    # scaling pixel value to (-1,1)
    img = np.array(img).astype(np.float32)
    img = ((img / 255.) - 0.5) * 2.0

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # result表示预测正确概率, 因为有两个分类，所以是一个有两个元素的列表，元素值分别表示是class_indices中此索引所表示分类的概率
    # 比如现在class_indeices中 两个索引 0: Masked; 1: unMasked
    # 那么result = [0.8XX, 0.1XX] 分别为 Masked 的概率 和 unMasked的概率
    result = np.squeeze(my_model.predict(img))
    # result = np.squeeze(model(img, training=False))
    # print(result)

    # numpy.argmax（result）会返回reuslt中最大值的索引，在这里返回0
    predict_class = np.argmax(result)

    # 这里解释了为什么需要class_indices
    # 因为DNN输出的只是一个表示概率的列表，它不会直接输出这张属于哪个分类，但是它会按照class_indices中的索引顺序来排列概率
    # 所以我们要做的就是在输出的列表中找到概率最大值的索引，然后去class_indices中找这个索引代表哪个分类
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
    #                                              result[predict_class])
    # plt.title(print_res)
    # print(print_res)
    # plt.show()

    result_class = class_indict[str(predict_class)]
    result_prob = result[predict_class]

    print(result_class)
    print(result_prob)

