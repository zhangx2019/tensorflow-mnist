import os

import input_data
import model
import tensorflow as tf

data = input_data.read_data_sets('MNIST_data',one_hot=True)

#create model
with tf.variable_scope('regression'): #进行命名操作
    x = tf.placeholder(tf.float32, [None,784])  #占位符(格式，张量)
    y, variables = model.regression(x)

#train
y_ = tf.placeholder('float',[None,10])
#为计算交叉熵，需添加一个新的占位符用于输入正确值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) #定义交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #预测
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #精确度

#参数先保存，再训练
saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000): #以随机梯度下降法来运行(1000次迭代)
        batch_xs,batch_ys = data.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys}) #feed_dict往里放数据

    print(sess.run(accuracy,feed_dict={x:data.test.images,y_:data.test.labels}))

#模型保存
    path = saver.save(
        sess,os.path.join(os.path.dirname(__file__),'data','regression.ckpt'),
        write_meta_graph=False,write_state=False)
    print('Saved:',path)
