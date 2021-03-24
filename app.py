import cv2
import numpy as np
from PIL import Image

from predict import load_model, predict_img


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


# 加载模型
print('模型加载中... 大概需要8-9s')
model = load_model()

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
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 将原图像的人脸部分剪切下来进行预测
    # cropped_img = cv_img[int(y):int(y + h), int(x):int(x + w)]
    cropped_img = cv_img[y:y + h, x:x + w]

    # 为了预测，将cv2格式的图像转为Image格式
    cap_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    result_class, result_prob = predict_img(img=cap_img, my_model=model)
    result_class_text = "Predict Result: " + result_class
    result_prob_text = "Accuracy: " + str(result_prob)

    # 读取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_text = "fps: " + str(fps)

    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, result_class_text, (120, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, result_prob_text, (360, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    cv2.imshow("cap", frame)

    if cv2.waitKey(100) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
