import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
minst=input_data.read_data_sets("./mnist/",one_hot=True)
one_minist=minst.train.images[800]
one_minist_img=one_minist.reshape((1,28,28,1))
# plt.imshow(one_minist_img)
# # plt.show()

train_x=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1],name="train_x")
train_y=tf.placeholder(dtype=tf.float32,shape=[None,10],name="train_y")
# 初始化参数
weights={
    'wc1': tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024,10],stddev=0.1)),
}
bias={
    'bc1':tf.Variable(tf.random_normal([64],stddev=0.1)),
    'bc2':tf.Variable(tf.random_normal([128],stddev=0.1)),
    'bd1':tf.Variable(tf.random_normal([1024],stddev=0.1)),
    'bd2':tf.Variable(tf.random_normal([10],stddev=0.1)),
}
# 卷积网络
conv1=tf.nn.conv2d(train_x,weights['wc1'],strides=[1,1,1,1],padding="SAME")
conv1=tf.nn.bias_add(conv1,bias['bc1'])
conv1=tf.nn.relu(conv1)
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

conv2=tf.nn.conv2d(pool1,weights['wc2'],strides=[1,1,1,1],padding="SAME")
conv2=tf.nn.bias_add(conv2,bias['bc2'])
conv2=tf.nn.relu(conv2)
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

reshape_pool2=tf.reshape(pool2,shape=[-1,7*7*128])
fc1=tf.matmul(reshape_pool2,weights['wd1'])
fc1=tf.nn.bias_add(fc1,bias['bd1'])
fc1=tf.nn.relu(fc1)

fc2=tf.matmul(fc1,weights['wd2'])
fc2=tf.nn.bias_add(fc2,bias['bd2'])
# fc2=tf.nn.relu(fc2)

# 模型优化器，求解
# cross_entropy = -tf.reduce_sum( train_y * tf.log(fc2) )
cross_entropy=tf.reduce_mean(tf.square(fc2-train_y))
train_step = tf.train.AdamOptimizer(0.001).minimize( cross_entropy )
correct_perd = tf.equal( tf.argmax(fc2, 1), tf.argmax(train_y,1) )
accuracy = tf.reduce_mean( tf.cast( correct_perd, "float" ) )
#
# loss=tf.reduce_mean(fc2-train_y)
# optimizer=tf.train.GradientDescentOptimizer(0.001)
# train_op=optimizer.minimize(loss)
save=tf.train.Saver()
initial= tf.global_variables_initializer()
sess=tf.Session()
sess.run(initial)
if __name__=="__main__":
    train=False
    if train:
        for i in range(300):
            batch_xx,batch_yy=minst.train.next_batch(100)
            batch_xx=batch_xx.reshape((-1,28,28,1))
            # print(batch_xx.shape)
            val,_=sess.run([accuracy,train_step],feed_dict={train_x:batch_xx,train_y:batch_yy})
            print("accuracy:\t",val)
        lk=save.save(sess,"./chekpoint/model.ckpt")
        # 保存模型
    else:
        model=save.restore(sess,"./chekpoint/model.ckpt")
        fc2=sess.run(fc2, feed_dict={train_x: one_minist_img})
        out=tf.argmax(fc2, 1)
        print(sess.run(out))
        one_minist_img=one_minist_img.reshape((28,28))
        plt.imshow(one_minist_img)
        plt.show()