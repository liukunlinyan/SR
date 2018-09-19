import tensorflow as tf
import matplotlib.pyplot as plt
sess=tf.Session()
learning_rate=0.001
groable=tf.Variable(tf.constant(0))
lrate=tf.train.exponential_decay(learning_rate,groable,100,0.89)
# optmi=tf.train.GradientDescentOptimizer(lrate)
x=[]
y=[]
for i in range(3000):
    lr=sess.run(lrate,{groable:i})
    x.append(i)
    y.append(lr)
plt.figure(1)
plt.plot(x, y, 'r-')  # staircase=False
plt.show()