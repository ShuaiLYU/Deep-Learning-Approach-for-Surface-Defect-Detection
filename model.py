import os
import numpy as np
import tensorflow as tf
slim=tf.contrib.slim
from config import  *
from tensorflow.python.ops import embedding_ops
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import math_ops

class Model(object):

    def __init__(self,sess,param):
        self.step = 0
        self.__session = sess
        self.is_training=True
        self.__learn_rate = param["learn_rate"]
        self.__learn_rate=param["learn_rate"]
        self.__max_to_keep=param["max_to_keep"]
        self.__checkPoint_dir = param["checkPoint_dir"]
        self.__restore = param["b_restore"]
        self.__mode= param["mode"]
        self.is_training=True
        self.__batch_size = param["batch_size"]
        if  self.__mode is "savaPb" :
            self.__batch_size = 1
        ################ Building graph
        with self.__session.as_default():
            self.build_model()
        ###############参数初始化，或者读入参数
        with self.__session.as_default():
            self.init_op.run()
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.__max_to_keep)
            # Loading last save if needed
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                if ckpt:
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)
                    print('Restoring from epoch:{}'.format( self.step))
                    self.step+=1

    def build_model(self):
        def SegmentNet(input, scope, is_training, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm):
                    net = slim.conv2d(input, 32, [5, 5],scope='conv1')
                    net = slim.conv2d(net, 32, [5, 5], scope='conv2')
                    net=slim.max_pool2d(net,[2,2],[2,2],scope='pool1')

                    net = slim.conv2d(net, 64, [5, 5],scope='conv3')
                    net = slim.conv2d(net, 64, [5, 5], scope='conv4')
                    net = slim.conv2d(net, 64, [5, 5], scope='conv5')
                    net=slim.max_pool2d(net,[2,2],[2,2],scope='pool2')

                    net = slim.conv2d(net, 64, [5, 5],scope='conv6')
                    net = slim.conv2d(net, 64, [5, 5], scope='conv7')
                    net = slim.conv2d(net, 64, [5, 5],scope='conv8')
                    net = slim.conv2d(net, 64, [5, 5], scope='conv9')
                    net=slim.max_pool2d(net,[2,2],[2,2],scope='pool3')

                    net = slim.conv2d(net, 1024, [15, 15], scope='conv10')
                    features=net
                    net = slim.conv2d(net, 1, [1, 1],activation_fn=None, scope='conv11')
                    logits_pixel=net
                    net=tf.sigmoid(net, name=None)
                    mask=net
            return features,logits_pixel,mask

        def DecisionNet(feature,mask, scope, is_training,num_classes=2, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm):
                    net=tf.concat([feature,mask],axis=3)
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool1')
                    net = slim.conv2d(net, 8, [5, 5], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool2')
                    net = slim.conv2d(net, 16, [5, 5], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool3')
                    net = slim.conv2d(net, 32, [5, 5], scope='conv3')
                    vector1=math_ops.reduce_mean(net,[1,2],name='pool4', keepdims=True)
                    vector2=math_ops.reduce_max(net,[1,2],name='pool5', keepdims=True)
                    vector3=math_ops.reduce_mean(mask,[1,2],name='pool6', keepdims=True)
                    vector4=math_ops.reduce_max(mask,[1,2],name='pool7', keepdims=True)
                    vector=tf.concat([vector1,vector2,vector3,vector4],axis=3)
                    vector=tf.squeeze(vector,axis=[1,2])
                    logits = slim.fully_connected(vector, num_classes,activation_fn=None)
                    output=tf.argmax(logits,axis=1)
                    return  logits,output

        Image = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0],IMAGE_SIZE[1], 1), name='Image')
        PixelLabel=tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0]/8,IMAGE_SIZE[1]/8, 1), name='PixelLabel')
        Label = tf.placeholder(tf.int32, shape=(self.__batch_size), name='Label')
        features, logits_pixel, mask=SegmentNet(Image,'segment',self.is_training)
        logits_class,output_class=DecisionNet(features,mask, 'decision', self.is_training)
        #损失函数
        logits_pixel=tf.reshape(logits_pixel,[self.__batch_size,-1])
        PixelLabel_reshape=tf.reshape(PixelLabel,[self.__batch_size,-1])
        loss_pixel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=PixelLabel_reshape))
        loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_class,labels=Label))
        loss_total=loss_pixel+loss_class
        optimizer = tf.train.GradientDescentOptimizer(self.__learn_rate)
        train_var_list = [v for v in tf.trainable_variables() ]
        train_segment_var_list = [v for v in tf.trainable_variables() if 'segment' in v.name ]
        train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]
        optimize_segment = optimizer.minimize(loss_pixel,var_list=train_segment_var_list)
        optimize_decision = optimizer.minimize(loss_class, var_list=train_decision_var_list)
        optimize_total = optimizer.minimize(loss_total, var_list=train_var_list)
        init_op=tf.global_variables_initializer()
        self.Image=Image
        self.PixelLabel = PixelLabel
        self.Label = Label
        self.features = features
        self.mask = mask
        self.logits_class=logits_class
        self.output_class=output_class
        self.loss_pixel = loss_pixel
        self.loss_class = loss_class
        self.loss_total = loss_total
        self.optimize_segment = optimize_segment
        self.optimize_decision = optimize_decision
        self.optimize_total = optimize_total
        self.init_op=init_op

    def save(self):
        self.__saver.save(
            self.__session,
            os.path.join(self.__checkPoint_dir, 'ckp'),
            global_step=self.step
        )

    # def save_PbModel(self):
    #     output_name=self.__decoded.op.name
    #     #output_name = self.__decoded.name.split(":")[0]
    #     input1_name=self.__inputs.name.split(":")[0]
    #     input2_name = self.__seq_len.name.split(":")[0]
    #     print("模型保存为pb格式，输入节点name：{}，{},输出节点name: {}".format(input1_name,input2_name,output_name))
    #     #constant_graph = graph_util.convert_variables_to_constants(self.__session, self.__session.graph_def, [output_name])
    #     constant_graph=graph_util.convert_variables_to_constants(self.__session,self.__session.graph_def,["SparseToDense"])
    #     with tf.gfile.GFile(self.__model_path+'Model.pb', mode='wb') as f:
    #         f.write(constant_graph.SerializeToString())
    #def PbModel(self):
        # with gfile.FastGFile('Model.pb', 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     sess.graph.as_default()
        #     tf.import_graph_def(graph_def, name='')  # 导入计算图
        #     for i, n in enumerate(graph_def.node):
        #         print("Name of the node - %s" % n.name)
		#
        # # 输入
        # input_x = sess.graph.get_tensor_by_name('Placeholder:0')
        # input_seq_len = sess.graph.get_tensor_by_name('seq_len:0')
        # # 输出
        # op = sess.graph.get_tensor_by_name('SparseToDense:0')