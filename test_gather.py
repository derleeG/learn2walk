import tensorflow as tf

batch_size = 3
sample_size = 2
a = tf.expand_dims(tf.range(6),1)

a = tf.concat([a, a, a, a], 1)

b = tf.random_uniform([tf.shape(a)[0],1])

b = tf.transpose(tf.reshape(b, [sample_size, -1]), [1, 0])
a = tf.transpose(tf.reshape(a, [sample_size, -1, 4]), [1, 0, 2])


d = tf.expand_dims(tf.argmax(b, axis=1),1)

d = tf.to_int32(d)
d = tf.concat([tf.expand_dims(tf.range(batch_size),1), d], axis=1)

        
c = tf.gather_nd(a, d)


sess = tf.Session()

print sess.run([a,b,d, c])