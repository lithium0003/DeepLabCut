'''
Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import re, os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.nnet import losses
vers = (tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF=tf.compat.v1
else:
    TF=tf

net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101,
             'resnet_152': resnet_v1.resnet_v1_152}


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with TF.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=2,
                                         scope='block4')
            return pred


def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2]
    }


class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_features(self, inputs):
        net_fun = net_funcs[self.cfg.net_type]

        mean = tf.constant(self.cfg.mean_pixel,
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        # The next part of the code depends upon which tensorflow version you have.
        vers = tf.__version__
        vers = vers.split(".") #Updated based on https://github.com/AlexEMG/DeepLabCut/issues/44
        if int(vers[0])==1 and int(vers[1])<4: #check if lower than version 1.4.
            with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16)
        else:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16,is_training=False)

        return net,end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg

        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        with TF.variable_scope('pose', reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_joints)
            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
            if cfg.intermediate_supervision:
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           cfg.num_joints)

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)

    def model_fn(self, features, labels, mode, params):
        tpu = os.getenv('TPU', default=False)
        cfg = self.cfg

        heads = self.get_net(features)
        predictions = {
            'part_prob': tf.sigmoid(heads['part_pred']),
            'locref': heads['locref']
        }

        if not tpu:
            TF.summary.image('input', features)
            show_label = tf.concat([
                features[:,:,:,:2] / 255.,
                tf.image.resize_bilinear(
                    tf.expand_dims(labels['part_score_targets'][:,:,:,0], axis=3), 
                    tf.shape(features)[1:3])],
                axis=3)
            TF.summary.image('label', show_label)
            show_pred = tf.image.resize_bilinear(
                    tf.expand_dims(predictions['part_prob'][:,:,:,0], axis=3), 
                    tf.shape(features)[1:3])
            TF.summary.image('predict', show_pred)

        if mode == tf.estimator.ModeKeys.PREDICT:
            if tpu:
                return TF.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)
            else:
                return TF.estimator.EstimatorSpec(mode, predictions=predictions)

        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = labels['part_score_weights'] if weigh_part_predictions else 1.0

        def add_part_loss(pred_layer):
            return TF.losses.sigmoid_cross_entropy(labels['part_score_targets'],
                                                   heads[pred_layer],
                                                   part_score_weights)

        loss = {}
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']
        if cfg.intermediate_supervision:
            loss['part_loss_interm'] = add_part_loss('part_pred_interm')
            total_loss = total_loss + loss['part_loss_interm']

        if cfg.location_refinement:
            locref_pred = heads['locref']
            locref_targets = labels['locref_targets']
            locref_weights = labels['locref_mask']

            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss['total_loss'] = total_loss

        if not tpu:
            for k, t in loss.items():
                TF.summary.scalar(k, t)
        metrics = {'loss': TF.metrics.mean(total_loss)}

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = TF.train.get_or_create_global_step()
 
            steps = cfg.multi_step
            vals = [s[0] for s in cfg.multi_step]
            boundaries = [s[1] for s in cfg.multi_step]
            vals += [vals[-1]]
            learning_rate = TF.train.piecewise_constant(global_step, boundaries, vals)

            if not tpu:
                TF.summary.scalar('learning_rate', learning_rate)

            if cfg.optimizer == "sgd":
                optimizer = TF.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            elif cfg.optimizer == "adam":
                optimizer = TF.train.AdamOptimizer(cfg.adam_lr)
            else:
                raise ValueError('unknown optimizer {}'.format(cfg.optimizer))

            if tpu:
                optimizer = TF.tpu.CrossShardOptimizer(optimizer)

            train_op = slim.learning.create_train_op(total_loss, optimizer)
        else:
            train_op = None


        if tpu:
            if mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(loss):
                    return {'loss': TF.metrics.mean(loss)}

                return TF.estimator.tpu.TPUEstimatorSpec(
                        mode=mode, loss=loss, eval_metrics=(metric_fn, [total_loss]))

            if mode == tf.estimator.ModeKeys.TRAIN:
                return TF.estimator.tpu.TPUEstimatorSpec(
                        mode=mode, loss=total_loss, train_op=train_op)
        else:
            return TF.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=total_loss,
                    train_op=train_op,
                    eval_metric_ops=metrics)
