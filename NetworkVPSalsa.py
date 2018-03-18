# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        ################################################
        self.obs = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='obs')
        self.obs2 = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='obs2')
        self.rew = tf.placeholder(tf.float32, [None], name='rew')
        self.term = tf.placeholder(tf.bool, [None], name="term")
        self.comact = tf.placeholder(tf.float32, [None, self.num_actions+6])
        self.com = self.comact[:,:6]
        self.act = self.comact[:,-self.num_actions:]

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')
        ###################################################

        # implement ddqn with salsa
        ###################################################

        # predict policy and value with observation
        obs_shape = self.obs.get_shape()
        obs_nb_elements = obs_shape[1] * obs_shape[2] * obs_shape[3]
        self.obs_pre = self._obs_pre(tf.reshape(self.obs, shape=[-1, obs_nb_elements._value]))
        self.obs2_pre = self._obs_pre(tf.reshape(self.obs2, shape=[-1, obs_nb_elements._value]))

        # self.obs_pre_com_1 = self._argmax_com(self.obs_pre, name="_1")
        # self.obs_pre_com_2 = self._argmax_com(self.obs_pre, name="_2")
        prob, ind, self.obs_pre_com_1 = self._pre_com(self.obs_pre)
        self.obs_pre_com_2 = self.obs_pre_com_1
        self.obs2_pre_com_1 = self._argmax_com(self.obs2_pre, name="_1")
        self.obs2_pre_com_2 = self._argmax_com(self.obs2_pre, name="_2")

        self.obs_pre_com_1_act = self._com_act(self.obs_pre, self.obs_pre_com_1)
        self.obs_pre_com_2_act = self._com_act(self.obs_pre, self.obs_pre_com_2)
        self.obs2_pre_com_1_act = self._com_act(self.obs2_pre, self.obs2_pre_com_1)
        self.obs2_pre_com_2_act = self._com_act(self.obs2_pre, self.obs2_pre_com_2)

        self.obs_pre_com_1_act_val_1 = self._act_val(self.obs_pre, self.obs_pre_com_1_act, "_1")
        self.obs_pre_com_2_act_val_2 = self._act_val(self.obs_pre, self.obs_pre_com_2_act, "_2")
        self.obs2_pre_com_1_act_val_2 = self._act_val(self.obs2_pre, self.obs2_pre_com_1_act, "_2")
        self.obs2_pre_com_2_act_val_1 = self._act_val(self.obs2_pre, self.obs2_pre_com_2_act, "_1")

        self.act_val_1 = self._act_val(self.obs_pre, self.act, '_1')
        self.act_val_2 = self._act_val(self.obs_pre, self.act, '_2')

        # predict joint rotation with observation and action
        self.act_env = self._act_env(self.obs_pre, self.act)
        self.env = (self.obs2[:,0,6:12,-1] - self.obs[:,0,6:12,-1])

        # predict joint rotation with observation and command
        self.com_act = self._com_act(self.obs_pre, self.com)
        self.com_act_env = self._act_env(self.obs_pre, self.com_act)


        # dual value network
        select_net = tf.random_uniform([])
        act_val_target1 = tf.where(self.term, self.rew, self.rew + tf.stop_gradient(self.obs2_pre_com_1_act_val_2)*Config.DISCOUNT)
        act_val_target2 = tf.where(self.term, self.rew, self.rew + tf.stop_gradient(self.obs2_pre_com_2_act_val_1)*Config.DISCOUNT)
        
        self.cost_v = tf.where(select_net > 0.5, tf.reduce_mean(self.l1_loss(self.act_val_1 - act_val_target1)),
                                                 tf.reduce_mean(self.l1_loss(self.act_val_2 - act_val_target2)))

        self.selected_action_prob = tf.reduce_sum(prob*ind, 1)

        self.cost_c_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * tf.where(select_net > 0.5, act_val_target1 - tf.stop_gradient(self.obs_pre_com_2_act_val_2),
                                                     act_val_target2 - tf.stop_gradient(self.obs_pre_com_1_act_val_1))
        self.cost_c_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.log(tf.maximum(prob, self.log_epsilon)) * prob, axis=1)
        
        self.cost_c_1_agg = tf.reduce_mean(self.cost_c_1, axis=0)
        self.cost_c_2_agg = tf.reduce_mean(self.cost_c_2, axis=0)
        self.cost_c = -(self.cost_c_1_agg + self.cost_c_2_agg)
        
        self.cost_e = -(self.correlation(self.act_env, self.env))
                      
        self.cost_a = -(self.correlation(self.com_act_env, self.com))

        theta_v = [var for var in self.graph.get_collection('trainable_variables') if 'act_val' in var.name]
        theta_c = [var for var in self.graph.get_collection('trainable_variables') if 'pre_com' in var.name]
        theta_a = [var for var in self.graph.get_collection('trainable_variables') if 'com_act' in var.name]
        theta_e = [var for var in self.graph.get_collection('trainable_variables') if 'act_env' in var.name]

        opt_v = tf.train.AdamOptimizer(learning_rate=1e-5)
        opt_c = tf.train.AdamOptimizer(learning_rate=1e-3)
        opt_a = tf.train.AdamOptimizer(learning_rate=1e-3)
        opt_e = tf.train.AdamOptimizer(learning_rate=1e-3)

        l2_v = 1e-3*tf.add_n([tf.nn.l2_loss(var) for var in theta_v])
        l2_c = 1e-3*tf.add_n([tf.nn.l2_loss(var) for var in theta_c])
        l2_a = 1e-3*tf.add_n([tf.nn.l2_loss(var) for var in theta_a])
        l2_e = 1e-3*tf.add_n([tf.nn.l2_loss(var) for var in theta_e])

        self.grads_and_vars_v = opt_v.compute_gradients(self.cost_v + l2_v, var_list=theta_v)
        self.grads_and_vars_c = opt_c.compute_gradients(self.cost_c + l2_c, var_list=theta_c)
        self.grads_and_vars_a = opt_a.compute_gradients(self.cost_a + l2_a + 2*self.cost_e + l2_e, var_list=theta_a+theta_e)
        # self.grads_and_vars_e = opt_e.compute_gradients(self.cost_e + l2_e, var_list=theta_e)

        self.grads_and_vars_v = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM), v) for g,v in self.grads_and_vars_v]
        self.grads_and_vars_c = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM), v) for g,v in self.grads_and_vars_c]
        self.grads_and_vars_a = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM), v) for g,v in self.grads_and_vars_a]
        # self.grads_and_vars_e = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM), v) for g,v in self.grads_and_vars_e]

        optimize_v = opt_v.apply_gradients(self.grads_and_vars_v, global_step=self.global_step)
        optimize_c = opt_c.apply_gradients(self.grads_and_vars_c, global_step=self.global_step)
        optimize_a = opt_a.apply_gradients(self.grads_and_vars_a, global_step=self.global_step)
        # optimize_e = opt_e.apply_gradients(self.grads_and_vars_e, global_step=self.global_step)

        self.train_op = [optimize_a, optimize_v, optimize_c]



    def _command_encode(self, x):
        return tf.to_float(tf.concat([tf.mod(x/3**i, 3) for i in range(6)], 1) - 1)

    def _command_decode(self, x):
        return tf.add_n([(x[:,i]+1)*3**i for i in range(6)])

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # summaries.append(tf.summary.scalar("Ccost_advantage", self.cost_c_1_agg))
        # summaries.append(tf.summary.scalar("Ccost_entropy", self.cost_c_2_agg))
        # summaries.append(tf.summary.scalar("Ccost", self.cost_c))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("Ecost", self.cost_e))
        summaries.append(tf.summary.scalar("Acost", self.cost_a))
        summaries.append(tf.summary.scalar("Ccost", self.cost_c))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_com", self.obs_pre_com_1))
        summaries.append(tf.summary.histogram("activation_act", self.obs_pre_com_1_act))
        summaries.append(tf.summary.histogram("activation_val_1", self.act_val_1))
        summaries.append(tf.summary.histogram("activation_val_2", self.act_val_2))
        summaries.append(tf.summary.histogram("activation_env", self.act_env))
        summaries.append(tf.summary.histogram("activation_prob", self.selected_action_prob))
        summaries.append(tf.summary.histogram("cost_c_1", self.cost_c_1_agg))
        summaries.append(tf.summary.histogram("cost_c_2", self.cost_c_2_agg))
        summaries.append(tf.summary.histogram("observation_env", self.env))
        for grads, var in self.grads_and_vars_v:
            summaries.append(tf.summary.histogram("gradient_v_%s" % var.name, grads))
        for grads, var in self.grads_and_vars_a:
            summaries.append(tf.summary.histogram("gradient_a_%s" % var.name, grads))
        for grads, var in self.grads_and_vars_c:
            summaries.append(tf.summary.histogram("gradient_c_%s" % var.name, grads))
        # for grads, var in self.grads_and_vars_e:
        #     summaries.append(tf.summary.histogram("gradient_e_%s" % var.name, grads))

        for i in range(6):
            summaries.append(tf.summary.histogram("com_"+str(i), self.com[:,i]))
    

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def l1_loss(self, x):
        return  tf.sqrt(tf.square(x)+1e-12)

    def correlation(self, x, y):
        # x = tf.contrib.layers.batch_norm(x, center=False, scale=True, is_training=Config.PLAY_MODE)
        # y = tf.contrib.layers.batch_norm(y, center=False, scale=True, is_training=Config.PLAY_MODE)
        x_mean, x_var = tf.nn.moments(x, [0])
        y_mean, y_var = tf.nn.moments(y, [0])
        x_mean = tf.stop_gradient(x_mean)
        x_var = tf.stop_gradient(x_var)
        y_mean = tf.stop_gradient(y_mean)
        y_var = tf.stop_gradient(y_var)
        x = (x-x_mean)/tf.sqrt(x_var+1e-3)
        y = (y-y_mean)/tf.sqrt(y_var+1e-3)

        return tf.reduce_mean(x*y)

        
    def _obs_pre(self, o):
        # x = self.dense_layer(o, 256, 'obs_pre1')
        # x = self.dense_layer(x, 256, 'obs_pre2')
        return o

    def _pre_com(self, p):
        x = self.dense_layer(p, 256, 'pre_com1')
        x = self.dense_layer(x, 256, 'pre_com2')
        x = self.dense_layer(x, 3**6, 'pre_com3', func=None)
        prob = (tf.nn.softmax(x) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * 3**6)
        if Config.PLAY_MODE:
            ind = tf.expand_dims(tf.argmax(prob, axis=1),1)
        else:
            ind = tf.multinomial(tf.log(prob), 1)
        com = self._command_encode(ind)
        ind = tf.to_float(tf.one_hot(tf.squeeze(ind, 1), 3**6))
        return prob, ind, com

    def _argmax_com(self, o, sample_size=128, name = ""):
        batch_size = tf.shape(o)[0]
        o = tf.tile(o, [sample_size,1])
        c = tf.to_float(tf.random_uniform([batch_size*sample_size, 6], -1, 2, dtype=tf.int32))
        a = self._com_act(o, c)
        v = self._act_val(o, a, name=name)
        v = tf.transpose(tf.reshape(v, [sample_size, -1]), [1, 0])
        c = tf.transpose(tf.reshape(c, [sample_size, -1, 6]), [1, 0, 2])
        if Config.PLAY_MODE or True:
            arg_max = tf.expand_dims(tf.argmax(v, axis=1),1)
        else:
            arg_max = tf.multinomial(v, 1)
        arg_max = tf.to_int32(arg_max)
        arg_max = tf.concat([tf.expand_dims(tf.range(batch_size),1), arg_max], axis=1)
        c = tf.gather_nd(c, arg_max)
        if not Config.PLAY_MODE:
            c = self.noisy(c, minval=-1)
        return c

    def _com_act(self, p, c):
        x = tf.concat([p, c], 1)
        x = self.dense_layer(x, 256, 'com_act1')
        x = self.dense_layer(x, 256, 'com_act2')
        x = self.dense_layer(x, 256, 'com_act3')
        x1 = self.dense_layer(x, self.num_actions, 'com_act4', func=tf.nn.sigmoid)
        x2 = self.dense_layer(x, self.num_actions, 'com_act5', func=tf.nn.sigmoid)
        if not Config.PLAY_MODE:
            x1 = self.noisy(x1)
            x2 = self.noisy(x2)
        return x1 * x2

    def _act_env(self, p, a):
        x = tf.concat([p, a], 1)
        x = self.dense_layer(x, 256, 'act_env1')
        x = self.dense_layer(x, 256, 'act_env2')
        x = self.dense_layer(x, 6, 'act_env3', func=tf.nn.tanh)
        return x

    def _act_val(self, p, a, name = ""):
        x = tf.concat([p, a], 1)
        x = self.dense_layer(x, 256, 'act_val1'+name)
        x = self.dense_layer(x, 256, 'act_val2'+name)
        x = self.dense_layer(x, 256, 'act_val3'+name)
        x = self.dense_layer(x, 1, 'act_val4'+name, func=None)
        return tf.squeeze(x, axis=[1])

    @staticmethod
    def selu(x):  
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

    def noisy(self, x, stddev=0.1, minval=0, maxval=1):
        return tf.clip_by_value(x + tf.random_normal(tf.shape(x), stddev=stddev), minval, maxval)
        
    def dense_layer(self, input, out_dim, name, func=tf.nn.relu, useRelu=True):
        if func is tf.nn.relu and useRelu:
            func = self.selu
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name) as scope:
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            try:
                w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
                b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
            except ValueError:
                scope.reuse_variables()
                w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
                b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
                
            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        if np.random.random_sample() > 0.5:
            prediction = self.sess.run(self.obs_pre_com_1_act_val_1, feed_dict={self.obs: x})
        else:
            prediction = self.sess.run(self.obs_pre_com_2_act_val_2, feed_dict={self.obs: x})
        return prediction

    def predict_p(self, x):
        if np.random.random_sample() > 0.5:
            com, act = self.sess.run([self.obs_pre_com_1, self.obs_pre_com_1_act], feed_dict={self.obs: x})
        else:
            com, act = self.sess.run([self.obs_pre_com_2, self.obs_pre_com_2_act], feed_dict={self.obs: x})
        prediction = np.concatenate([com, act], axis=1)
        return prediction
    
    def predict_p_and_v(self, x):
        if np.random.random_sample() > 0.5:
            com, act, val = self.sess.run([self.obs_pre_com_1, self.obs_pre_com_1_act,
                                    self.obs_pre_com_1_act_val_1], feed_dict={self.obs: x})
        else:
            com, act, val = self.sess.run([self.obs_pre_com_2, self.obs_pre_com_2_act,
                                    self.obs_pre_com_2_act_val_2], feed_dict={self.obs: x})
        prediction = np.concatenate([com, act], axis=1)
        return prediction, val
    
    def train(self, o, r, a, n, d, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.obs: o, self.rew: r, self.comact: a, self.obs2: n, self.term:d})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, o, r, a, n, d):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.obs: o, self.rew: r, self.comact: a, self.obs2: n, self.term:d})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
