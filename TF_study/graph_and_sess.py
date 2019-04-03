"""
For practice

O'REILLY
ML with scikit-learn and TensorFlow

section9
p.227~

2019/3/19 16:00
"""

import tensorflow as tf

# only making cal graph. not run, not init val yet.
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2


# launch a session
# sess = tf.Session()
#
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
#
# sess.close

# better
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()


# simplify initialization
init = tf.global_variables_initializer()  # init node

with tf.Session() as sess:
    init.run()
    result = f.eval()

print(result)


# manage multi graphs
print(x.graph is tf.get_default_graph())

g = tf.Graph()
with g.as_default():
    x2 = tf.Variable(2, name="x2")
    print(x.graph is tf.get_default_graph())
    print(x2.graph is tf.get_default_graph())

print(x.graph is tf.get_default_graph())
print(x2.graph is tf.get_default_graph())