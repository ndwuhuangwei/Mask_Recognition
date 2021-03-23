import cv2
import numpy
from PIL import Image

from predict import load_model, predict_img

# 加载模型
print('模型加载中... 大概需要8-9s')
model = load_model()

# 0:笔记本内置摄像头， 1:usb设想头
cap = cv2.VideoCapture(0)


while 1:
    # ret取值为True或False, 表示当前是否有读取到图片
    # frame代表当前截取一帧的图片，颜色空间为RGB（ cv2.imread(path)读取的颜色空间是BGR ）, type是np.array
    ret, frame = cap.read()
    if ret is False:
        print('No image detected')
        break

    # 将cv2.VedioCapture读取到的np.arrary模式的图片转换成image格式
    cap_img = Image.fromarray(frame)
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
