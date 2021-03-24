# 2021-3-24更新
感谢 电子信息学院石雅琪同学的建议，先检测人脸再对人脸作预测可以有效避免环境干扰

本程序目前存在一个bug,在检测不到人脸时画面会卡住，理论上它应该要在画面中实时显示错误信息，单步调试时功能完好，但是运行起来却会卡住。目前尚未找到解决方法。
<br>虽然没检测到人脸时会被卡住，但是一旦再次检测到人脸又会正常运行,所以此bug只是会影响用户体验，并不影响口罩识别的核心功能

# 项目简介
这是一个人脸口罩识别的小项目

用的是经典的 mobilenetv2网络，能够调用电脑的摄像头实时判断摄像头前的人是否配套口罩<br>
图像上方会实时显示当前摄像头的帧率、预测的结果（摄像头里的人带不带口罩）以及预测的准确率

# 环境
python 3.7<br>
tensorflow 2.1.0 (至少2以上)<br>
opencv3.4.2

本项目使用于 win10<br>
推荐使用 pycharm + anaconda3 配置环境

# Quick Start
运行 app.py

在模型加载完成后（需要一点时间），程序就会开始运行

#如何换模型进行预测
本项目目前训练了两个模型，训练后的模型权重值（ckpt文件）全部位于save_weights中
> resMobileNetV2.ckpt.data-0000-of-0001
>
> resMobileNetV2.ckpt.index

> resMobileNetV2.ckpt.data-0000-of-0001
>
> resMobileNetV2.ckpt.index

除了checkpoint之外，每个模型都有两个文件，一个是XXX.ckpt.data-0000-of-0001 和 XXX.ckpt.index

如果要修改预测所需的模型:<br>
在 predict.py 中修改 weights_path 的值

注意 修改时应该写 resMobileNetV2.ckpt 或是 resMobileNetV2.ckpt<br>
其余后缀不用加
<br>
<br>

注意，预测时的模型搭建必须与训练时一样
```C
feature = MobileNetV2(num_classes=num_classes, include_top=False)
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(num_classes),
                             tf.keras.layers.Softmax()])
...
assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
model.load_weights(weights_path)

```

# 如何用新的数据集训练
因为数据集太大了，因此我没有在github上放数据集 <br>
你可以参考这里的数据集 https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

如果你要训练自己的数据集，首先在工程目录下新建一个data_set文件夹 <br>
新的数据集应该放入data_set中，目录结构如下：
> data_set
>> ...
>>
>> data_root
>>> your_data_set_dir
>>>> calss_1
>>>>
>>>> class_2
>>>>
>>>> ...

首先修改 class_indices.json 中的分类
<br>
<br>

然后，需要划分训练集和数据集，在 split_data.py 中修改 my_data_root 为上述结构中的 data_root，
修改 my_origin_face_path 为上述结构中的 your_data_set_dir<br>

运行 split_data.py, 会多出 train 和 val 两个文件夹，得到的目录应该如下所示：
> data_set
>> ...
>>
>> data_root
>>> your_data_set_dir
>>>> calss_1
>>>>
>>>> class_2
>>>>
>>>> ...
>>> 
>>> train
>>>>calss_1
>>>>
>>>> class_2
>>>>
>>>> ...
>>> 
>>> val
>>>> calss_1
>>>>
>>>> class_2
>>>>
>>>> ...
>>> 

顾名思义，train 文件夹存放着训练集，val 文件夹存放着验证集

默认分10% 的数据作为验证集，如果想修改，修改 main函数中的 split_rate
<br>
<br>

接着，在train_mobilenet_v2.py中修改如下值：
1. 修改 img_data_dir 为上述结构中的data_root
2. 修改 save_weights_path 为你想要保存的权重文件路径

本项目基于迁移学习的方法进行训练，默认是采用tensorlfow官网提供的预训练模型，此模型位于
tf_mobilenet_weights 文件夹中，如果想要使用自己的预训练模型，
在 train_mobilenet_v2.py 的main函数中修改 pre_weights_path 

# 运行情况展示
![avator](./Doc Pictures/run_1.png)
<img src="./Doc Pictures/run_1.png"/>

![avator](./Doc Pictures/run_2.png)
<img src="./Doc Pictures/run_2.png">

# 其他
1. 运行程序时可能会报错 Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
<br> 这是因为tensorflow的版本不对，但是无伤大雅，不影响程序运行

2. Doc文件夹中存放着本项目的问题日志, 如果程序运行时发生了什么问题可以去搜一下


# 作者介绍
吴黄巍<br>四川大学电子信息学院18级 