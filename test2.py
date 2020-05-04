import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras import datasets, optimizers

params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir="./model/tf_savedmodel", conversion_params=params)
# 完成转换,但是此时没有进行优化,优化在执行推理时完成
converter.convert()
converter.save('./model/trt_savedmodel')

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.datasets import mnist
import time
import cv2
import numpy as np

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.astype('float32')
x_test = x_test.reshape(10000, 784)
x_test /= 255

# 读取模型
saved_model_loaded = tf.saved_model.load("./model/trt_savedmodel", tags=[trt.tag_constants.SERVING])
# 获取推理函数,也可以使用saved_model_loaded.signatures['serving_default']
graph_func = saved_model_loaded.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# 将模型中的变量变成常量,这一步可以省略,直接调用graph_func也行
frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)

count = 20
for x, y in zip(x_test, y_test):
    x = tf.cast(x, tf.float32)
    start = time.time()
    # frozen_func(x)返回值是个列表
    # 列表中含有一个元素，就是输出tensor，使用.numpy()将其转化为numpy格式
    output = frozen_func(x)[0].numpy()
    end = time.time()
    times = (end - start) * 1000.0
    print("tensorrt times: ", times, " ms")
    result = np.argmax(output, 1)
    print("prediction result: ", result, "  |  ", "true result: ", y)

    if count == 0:
        break
    count -= 1