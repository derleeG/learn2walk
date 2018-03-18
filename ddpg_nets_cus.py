import tensorflow as tf
import numpy as np

def hist_summaries(*args):
  return tf.summary.merge([tf.summary.histogram(t.name,t) for t in args])

def fanin_init(shape,fanin=None):
  fanin = fanin or shape[0]
  v = 1/np.sqrt(fanin)
  return tf.random_uniform(shape, minval=-v, maxval=v)

def selu(x, name):  
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x), name)

# l1 = 400 # dm 400
# l2 = 300 # dm 300
lp = [128, 256, 256]
lv = [128, 256, 256]
lc = [128, 256, 256]
le = [128, 256, 256]

def theta_p(dimO, dimC):
  with tf.variable_scope("theta_p"):
    dimIn = dimO[0]
    dimOut = dimC[0]
    l = lp
    return [tf.Variable(fanin_init([dimIn, l[0]]), name='1w'),
            tf.Variable(fanin_init([l[0]], dimIn), name='1b'),
            tf.Variable(fanin_init([l[0], l[1]]), name='2w'),
            tf.Variable(fanin_init([l[1]], l[0]), name='2b'),
            tf.Variable(fanin_init([l[1], l[2]]), name='3w'),
            tf.Variable(fanin_init([l[2]], l[1]), name='3b'),
            tf.Variable(tf.random_uniform([l[2], dimOut], -3e-3, 3e-3), name='4w'),
            tf.Variable(tf.random_uniform([dimOut], -3e-3, 3e-3), name='4b')]

def noisy(x, stddev, minval, maxval):
  return tf.clip_by_value(x + tf.random_normal(tf.shape(x), stddev=stddev), minval, maxval)

def l1_loss(x):
  return  tf.sqrt(tf.square(x)+1e-12)
    
def policy(obs, theta, name='policy'):
  with tf.variable_scope(name, name, [obs]):
    h0 = tf.identity(obs, name='h0-obs')
    h1 = selu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
    h2 = selu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
    h3 = selu(tf.matmul(h2, theta[4]) + theta[5], name='h3')
    command = tf.nn.tanh(tf.matmul(h3, theta[6]) + theta[7], name='h4-command')

    summary = hist_summaries(h0, h1, h2, h3, command)
    return command, summary


def theta_v(dimO, dimA):
  dimIn = dimO[0] + dimA[0]
  dimOut = 1
  l = lv
  with tf.variable_scope("theta_v"):
    return [tf.Variable(fanin_init([dimIn, l[0]]), name='1w'),
            tf.Variable(fanin_init([l[0]], dimIn), name='1b'),
            tf.Variable(fanin_init([l[0], l[1]]), name='2w'),
            tf.Variable(fanin_init([l[1]], l[0]), name='2b'),
            tf.Variable(fanin_init([l[1], l[2]]), name='3w'),
            tf.Variable(fanin_init([l[2]], l[1]), name='3b'),
            tf.Variable(tf.random_uniform([l[2], dimOut], -3e-3, 3e-3), name='4w'),
            tf.Variable(tf.random_uniform([dimOut], -3e-3, 3e-3), name='4b')]
    
def value(obs, act, theta, name="value"):
  with tf.variable_scope(name, name, [obs, act]):
    h0o = tf.identity(obs, name='h0-obs')
    h0a = tf.identity(act, name='h0-act')
    h0 = tf.concat([h0o, h0a], 1)
    h1 = selu( tf.matmul(h0, theta[0]) + theta[1], name='h1')
    h2 = selu( tf.matmul(h1, theta[2]) + theta[3], name='h2')
    h3 = selu( tf.matmul(h2, theta[4]) + theta[5], name='h3')
    h4 = tf.matmul(h3, theta[6]) + theta[7]
    # value = tf.identity(h4, name='h4-value')
    value = tf.squeeze(h4, [1], name='h4-value')
    
    summary = hist_summaries(h0o, h0a, h1, h2, h3, value)
    return value, summary

def theta_c(dimO, dimC, dimA):
  dimIn = dimC[0] + dimO[0] 
  dimOut = dimA[0]
  l = lc
  with tf.variable_scope("theta_c"):
    return [tf.Variable(fanin_init([dimIn, l[0]]), name='1w'),
            tf.Variable(fanin_init([l[0]], dimIn), name='1b'),
            tf.Variable(fanin_init([l[0], l[1]]), name='2w'),
            tf.Variable(fanin_init([l[1]], l[0]), name='2b'),
            tf.Variable(fanin_init([l[1], l[2]]), name='3w'),
            tf.Variable(fanin_init([l[2]], l[1]), name='3b'),
            tf.Variable(tf.random_uniform([l[2], dimOut], -3e-3, 3e-3), name='4w-gate'),
            tf.Variable(tf.random_uniform([dimOut], -3e-3, 3e-3), name='4b-gate'),
            tf.Variable(tf.random_uniform([l[2], dimOut], -3e-3, 3e-3), name='4w-value'),
            tf.Variable(tf.random_uniform([dimOut], -3e-3, 3e-3), name='4b-value')]
    
def control(obs, com, theta, name="control"):
  with tf.variable_scope(name, name, [com]):
    h0o  = tf.identity(obs, name='h0-obs')
    h0c  = tf.identity(com, name='h0-com')
    h0 = tf.concat([h0o, h0c], 1)
    h1 = selu( tf.matmul(h0, theta[0]) + theta[1], name='h1')
    h2 = selu( tf.matmul(h1, theta[2]) + theta[3], name='h2')
    h3 = selu( tf.matmul(h2, theta[4]) + theta[5], name='h3')
    h4gate  = tf.nn.sigmoid(tf.matmul(h3, theta[6]) + theta[7], name='h4-gate')
    h4value  = tf.nn.sigmoid(tf.matmul(h3, theta[8]) + theta[9], name='h4-value')
    action = tf.identity(h4gate*h4value, name='h4-action')
    
    summary = hist_summaries(h0o, h0c, h1, h2, h3, h4gate, h4value, action)
    return action, summary, h4gate, h4value

def theta_e(dimO, dimA, dimR):
  dimIn = dimO[0] + dimA[0]
  dimOut = dimR[0]
  l = le
  with tf.variable_scope("theta_e"):
    return [tf.Variable(fanin_init([dimIn, l[0]]), name='1w'),
            tf.Variable(fanin_init([l[0]], dimIn), name='1b'),
            tf.Variable(fanin_init([l[0], l[1]]), name='2w'),
            tf.Variable(fanin_init([l[1]], l[0]), name='2b'),
            tf.Variable(fanin_init([l[1], l[2]]), name='3w'),
            tf.Variable(fanin_init([l[2]], l[1]), name='3b'),
            tf.Variable(tf.random_uniform([l[2], dimOut], -3e-3, 3e-3), name='4w'),
            tf.Variable(tf.random_uniform([dimOut], -3e-3, 3e-3), name='4b')]
    
def environment(obs, act, theta, name="environment"):
  with tf.variable_scope(name, name, [obs, act]):
    h0o = tf.identity(obs, name='h0-obs')
    h0a = tf.identity(act, name='h0-act')
    h0 = tf.concat([h0o, h0a], 1)
    h1 = selu( tf.matmul(h0, theta[0]) + theta[1], name='h1')
    h2 = selu( tf.matmul(h1, theta[2]) + theta[3], name='h2')
    h3 = selu( tf.matmul(h2, theta[4]) + theta[5], name='h3')
    rotation  = tf.nn.tanh(tf.matmul(h3, theta[6]) + theta[7], name='h4-rotation')
    
    summary = hist_summaries(h0o, h0a, h1, h2, h3, rotation)
    return rotation, summary
