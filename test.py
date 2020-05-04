import tensorflow as tf
from tensorflow.keras import datasets, optimizers


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.expand_dims(x, axis=-1)
    x = tf.cast(x, dtype=tf.float32) / 255.
    # x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

batchsz = 128


def train():
    # 可以直接使用datasets.mnist.load_data()，如果网络好，可以连接外网，
    # 如果下载不了，可以自己先下载文件
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    print('datasets:', x.shape, y.shape, x.min(), x.max())

    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(10000).batch(batchsz)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batchsz)

    # sample = next(iter(db))
    # print(sample[0].shape, sample[1].shape)
    inputs = tf.keras.Input(shape=(28, 28, 1), name='input')
    # [28, 28, 1] => [28, 28, 64]
    input = tf.keras.layers.Flatten(name="flatten")(inputs)
    fc_1 = tf.keras.layers.Dense(512, activation='relu', name='fc_1')(input)
    fc_2 = tf.keras.layers.Dense(256, activation='relu', name='fc_2')(fc_1)
    pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_2)

    model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
    model.summary()
    Loss = []
    Acc = []
    optimizer = optimizers.Adam(0.001)
    # epoches = 5
    for epoch in range(1):
        # 创建用于测试精度的参数
        total_num = 0
        total_correct = 0
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:

                pred = model(x)
                loss = tf.keras.losses.categorical_crossentropy(y_pred=pred,
                                                                y_true=y,
                                                                from_logits=False)
                loss = tf.reduce_mean(loss)
                grades = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grades, model.trainable_variables))
                # 输出loss值
            if step % 10 == 0:
                print("epoch: ", epoch, "step: ", step, "loss: ", loss.numpy())
                Loss.append(loss)

        # 计算精度，将全连接层的输出转化为概率值输出
        for step, (x_val, y_val) in enumerate(ds_val):
            # 预测测试集的输出

            pred = model(x_val)
            # pred = tf.nn.softmax(pred, axis=1)
            pred = tf.argmax(pred, axis=1)
            pred = tf.cast(pred, tf.int32)
            y_val = tf.argmax(y_val, axis=1)
            y_val = tf.cast(y_val, tf.int32)
            correct = tf.equal(pred, y_val)
            correct = tf.cast(correct, tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total_num += x_val.shape[0]
            if step % 20 == 0:
                acc_step = total_correct / total_num
                print("第" + str(step) + "步的阶段精度是：", acc_step)
                Acc.append(float(acc_step))

        acc = total_correct / total_num
        print("epoch %d test acc: " % epoch, acc)
    # 方式1：
    model.save('./model/tf_savedmodel', save_format='tf')
    # 方式2：
    # tf.saved_model.save(obj=model, export_dir="./model/")


if __name__ == "__main__":
    train()