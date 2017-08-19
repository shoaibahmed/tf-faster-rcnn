# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

import nets.inception_resnet_v2
from nets.inception_resnet_v2 import inception_resnet_v2_base
from nets.inception_resnet_v2 import block8

def inception_arg_scope(is_training=True,
                     weight_decay=0.00004,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=0.001,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': cfg.TRAIN.BN_TRAIN and is_training,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': cfg.TRAIN.BN_TRAIN,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class inc_res_v2(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)
    self._inc_res_v2_scope = 'InceptionResnetV2'

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  # def _build_base(self):
  #   with tf.variable_scope(self._resnet_scope, self._resnet_scope):
  #     net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
  #     net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
  #     net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

  #   return net

  def _build_network(self, sess, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # Preaux logits: 17x17
    # Skipped mixed_7a which was used for pooling (used ROI pool instead)
    with slim.arg_scope(inception_arg_scope(is_training=False)):
      net_conv, end_points = inception_resnet_v2_base(inputs=self._image, 
                                                      final_endpoint='PreAuxLogits', 
                                                      align_feature_maps=True,
                                                      scope=self._inc_res_v2_scope)


    self._act_summaries.append(net_conv)
    self._layers['head'] = net_conv
    with tf.variable_scope(self._inc_res_v2_scope, self._inc_res_v2_scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      rois = self._region_proposal(net_conv, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError

    with slim.arg_scope(inception_arg_scope(is_training=is_training)):
      # fc7, _ = resnet_v1.resnet_v1(pool5,
      #                              blocks[-1:],
      #                              global_pool=False,
      #                              include_root_block=False,
      #                              scope=self._resnet_scope)

      print (pool5.get_shape())
      net = slim.repeat(pool5, 9, block8, scale=0.20)
      net = block8(net, activation_fn=None)

      # 8 x 8 x 1536
      fc7 = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')

    with tf.variable_scope(self._inc_res_v2_scope, self._inc_res_v2_scope):
      # average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      print (fc7.get_shape())
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                        initializer, initializer_bbox)

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._inc_res_v2_scope + '/Conv2d_1a_3x3/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Converting first Convolution layer from RGB to BGR')
    with tf.variable_scope('Fix_Inception_layers') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        # conv1_rgb = tf.get_variable("Conv2d_1a_3x3_rgb", [7, 7, 3, 64], trainable=False)
        conv1_rgb = tf.get_variable("Conv2d_1a_3x3_rgb", [3, 3, 3, 32], trainable=False)
        restorer_fc = tf.train.Saver({self._inc_res_v2_scope + "/Conv2d_1a_3x3/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._inc_res_v2_scope + '/Conv2d_1a_3x3/weights:0'], 
                           tf.reverse(conv1_rgb, [2])))
