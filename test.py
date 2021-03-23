# from predict import load_model, predict_img
# import os
# from PIL import Image
# import numpy as np
# import json
# import glob
# import tensorflow as tf
# from predict import load_model
#
# my_model = load_model()
#
# my_img_dir = "./test_data"
# im_height = 224
# im_width = 224
# num_classes = 2
#
# for test_img in os.listdir(my_img_dir):
#     print(test_img)
#     img_path = os.path.join(my_img_dir, test_img)
#
#     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     img = Image.open(img_path)
#
#     # resize image to 224x224
#     img = img.resize((im_width, im_height))
#     # plt.imshow(img)
#
#     # scaling pixel value to (-1,1)
#     img = np.array(img).astype(np.float32)
#     img = ((img / 255.) - 0.5) * 2.0
#
#     # Add the image to a batch where it's the only member.
#     img = (np.expand_dims(img, 0))
#
#     # read class_indict
#     json_path = './class_indices.json'
#     assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
#
#     json_file = open(json_path, "r")
#     class_indict = json.load(json_file)
#
#     # result表示预测正确概率, 因为有两个分类，所以是一个有两个元素的列表，元素值分别表示是class_indices中此索引所表示分类的概率
#     # 比如现在class_indeices中 两个索引 0: Masked; 1: unMasked
#     # 那么result = [0.8XX, 0.1XX] 分别为 Masked 的概率 和 unMasked的概率
#     result = np.squeeze(my_model.predict(img))
#     # result = np.squeeze(model(img, training=False))
#     # print(result)
#
#     # numpy.argmax（result）会返回reuslt中最大值的索引，在这里返回0
#     predict_class = np.argmax(result)
#
#     # 这里解释了为什么需要class_indices
#     # 因为DNN输出的只是一个表示概率的列表，它不会直接输出这张属于哪个分类，但是它会按照class_indices中的索引顺序来排列概率
#     # 所以我们要做的就是在输出的列表中找到概率最大值的索引，然后去class_indices中找这个索引代表哪个分类
#     # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
#     #                                              result[predict_class])
#     # plt.title(print_res)
#     # print(print_res)
#     # plt.show()
#
#     result_class = class_indict[str(predict_class)]
#     result_prob = result[predict_class]
#
#     print(result_class)
#     print(result_prob)

# import cv2
# import numpy as np
# # from tkinter import messagebox
# # print("这是一个弹出提示框")
# # messagebox.showinfo("提示","我是一个提示框")


# def face_detect(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_detector = cv2.CascadeClassifier("C:/Users/Whw03/Anaconda3/envs/github_test/Library/etc/haarcascades/"
#                                           "haarcascade_frontalface_default.xml")
#     # 最后一个参数设为1，会有两个框，faces会是一个二维列表，其中第一个列表元素代表的框是正确的
#     # 设为2，只会留下不正确的偏小的那个框
#     faces = face_detector.detectMultiScale(gray, 1.1, 1)
#     # useful_face = faces[:1]
#     # x = faces[:1, 0][0]
#     # y = faces[:1, 1][0]
#     # w = faces[:1, 2][0]
#     # h = faces[:1, 3][0]
#     x = faces[0][0]
#     # y = faces[0]
#     # w = faces[0]
#     # h = faces[0]
#     # return x, y, w, h
#     return int(x)
#     # x, y为左上角，x为横轴，y为纵轴，原点在整张图片的左上角
#     # for x, y, w, h in useful_face:
#     #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     # cv2.imshow("result", image)
#
#
# img = cv2.imread("./test_Data/test1.jpg")
# # x, y, w, h = face_detect(img)
# x = face_detect(img)
# print(type(x))
# print(str(x))
# # print(str(y))
# # print(str(w))
# # print(str(h))
# # y[y0:y1, x0:x1]
# # cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
# # cv2.imshow("result", cropped_img)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


import cv2
import numpy as np
from PIL import Image

# from predict import load_model, predict_img


def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier("C:/Users/Whw03/Anaconda3/envs/github_test/Library/etc/haarcascades/"
                                          "haarcascade_frontalface_default.xml")
    # 最后一个参数设为1，会有两个框，faces会是一个二维列表，其中第一个列表元素代表的框是正确的
    # 设为2，只会留下不正确的偏小的那个框
    faces = face_detector.detectMultiScale(gray, 1.1, 1)
    print(faces)
    # useful_face = faces[:1]
    # x, y为左上角，x为横轴，y为纵轴，原点在整张图片的左上角
    # for x, y, w, h in faces[:1]:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # cv2.imshow("result", image)
    # print(len(faces))
    if len(faces):
        xi = faces[0][0]
        yi = faces[0][1]
        wi = faces[0][2]
        hi = faces[0][3]
        return int(xi), int(yi), int(wi), int(hi)
    else:
        return 0, 0, 0, 0


# # 加载模型
# print('模型加载中... 大概需要8-9s')
# model = load_model()

# 0:笔记本内置摄像头， 1:usb设想头
cap = cv2.VideoCapture(0)

warning_text = "No Face Detected"

while 1:
    # ret取值为True或False, 表示当前是否有读取到图片
    # frame代表当前截取一帧的图片，颜色空间为RGB（ cv2.imread(path)读取的颜色空间是BGR ）, type是np.array
    ret, frame = cap.read()
    if ret is False:
        print('No image detected')
        break

    # 将cv2.VedioCapture读取到的np.arrary模式的图片转换成image格式
    # cap_img = Image.fromarray(frame)

    # 为了进行人脸检测，将摄像头读取到的颜色空间RGB转换为BGR
    cv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x, y, w, h = face_detect(cv_img)

    # 如果没有检测到人脸，这个frame就丢弃
    if x == 0 and y == 0 and w == 0 and h == 0:
        cv2.putText(cv_img, warning_text, (100, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow("cap", cv_img)
        continue

    # 将实时图像中的人脸框出来
    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # # 将原图像的人脸部分剪切下来进行预测
    # # cropped_img = cv_img[int(y):int(y + h), int(x):int(x + w)]
    # cropped_img = cv_img[y:y + h, x:x + w]
    #
    # # 为了预测，将cv2格式的图像转为Image格式
    # cap_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    #
    # result_class, result_prob = predict_img(img=cap_img, my_model=model)
    # result_class_text = "Predict Result: " + result_class
    # result_prob_text = "Accuracy: " + str(result_prob)
    #
    # # 读取帧率
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fps_text = "fps: " + str(fps)
    #
    # cv2.putText(cv_img, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    # cv2.putText(cv_img, result_class_text, (120, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    # cv2.putText(cv_img, result_prob_text, (360, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    cv2.imshow("cap", cv_img)

    if cv2.waitKey(100) & 0xff == ord('q'):
        break

