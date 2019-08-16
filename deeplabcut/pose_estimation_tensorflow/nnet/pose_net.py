'''
Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import re, os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.nnet import losses
import tensorflow as tf
vers = (tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF=tf.compat.v1
else:
    TF=tf
import numpy as np

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

    def extract_predictions(self, heads):
        with tf.name_scope('extract_predictions') as scope:
            cfg = self.cfg
            part_prob = tf.sigmoid(heads['part_pred'])
            if cfg.location_refinement:
                locref = heads['locref']
                locref_shape = tf.shape(locref)
                locref = tf.reshape(locref, [locref_shape[0], locref_shape[1], locref_shape[2], -1, 2])
                locref *= cfg.locref_stdev
            else:
                locref = None

            scmap_shape = tf.shape(part_prob)
            scmap = part_prob
            #batch x ny x nx x num_joints
            if self.num_outputs == 1:
                scmap_flat = tf.reshape(part_prob, [scmap_shape[0], -1, scmap_shape[3]])
                scmap_top = tf.expand_dims(tf.math.argmax(scmap_flat, axis=1), axis=2)
                scmap_top = tf.cast(scmap_top, tf.int32)
                #batch x num_joints x num_outputs
            else:
                scmap_flat = tf.manip.reshape(part_prob, [scmap_shape[0], -1, scmap_shape[3]])
                scmap_flat = tf.transpose(scmap_flat, [0, 2, 1])
                top_values, scmap_top = tf.math.top_k(scmap_flat, self.num_outputs)
                scmap_top = tf.cast(scmap_top, tf.int32)
                #batch x num_joints x num_outputs

            locref_shape = tf.shape(locref)
            locref = tf.reshape(locref, [locref_shape[0], -1, locref_shape[3], 2])
            #batch x ny*nx x num_joints x 2
            locref = tf.transpose(locref, [0, 2, 3, 1])
            #batch x num_joints x 2 x ny*nx
            locref_x, locref_y = tf.unstack(locref, axis=2) 
            #batch x num_joints x ny*nx
                    
            scmap = tf.reshape(scmap, [scmap_shape[0], -1, scmap_shape[3]])
            #batch x ny*nx x num_joints
            scmap = tf.transpose(scmap, [0, 2, 1])
            #batch x num_joints x ny*nx

            data_shape = tf.shape(locref_x)
            index_base = tf.range(data_shape[0]*data_shape[1]*data_shape[2])
            index_base = tf.reshape(index_base, data_shape)
            index_base = tf.tile(tf.reduce_min(index_base, axis=2, keepdims=True),[1,1,self.num_outputs])
            #batch x num_joints x num_outputs

            locref_x = tf.gather(tf.reshape(locref_x,[-1]), index_base+scmap_top)
            locref_y = tf.gather(tf.reshape(locref_y,[-1]), index_base+scmap_top)
            P = tf.gather(tf.reshape(scmap,[-1]), index_base+scmap_top)
            #batch x num_joints x num_outputs

            scmap_top_shape = tf.shape(scmap_top)
            def unravel_index(indices, shape):
                with tf.name_scope('unravel_index'):
                    indices = tf.expand_dims(indices, 0)
                    shape = tf.expand_dims(shape, 1)
                    strides = tf.math.cumprod(shape, reverse=True)
                    strides_shifted = tf.math.cumprod(shape, exclusive=True, reverse=True)
                    return (indices // strides_shifted) % strides
            Y,X = tf.unstack(unravel_index(tf.reshape(scmap_top,[-1]), self.sm_size), axis=0)
            Y = tf.reshape(Y, scmap_top_shape)
            X = tf.reshape(X, scmap_top_shape)
            #batch x num_joints x num_outputs

            X = tf.cast(X, tf.float32)*cfg.stride + .5*cfg.stride + locref_x
            Y = tf.cast(Y, tf.float32)*cfg.stride + .5*cfg.stride + locref_y
            #batch x num_joints x num_outputs

            return X, Y, P

    def make_colormap(self, num):
        def color_jet(v, dv):
            dv = tf.cast(dv, tf.float32) / 4
            v = tf.cast(v, tf.float32)
            return tf.case({
                tf.less(v, dv): lambda: ((tf.constant(0, tf.float32), v / dv, tf.constant(1, tf.float32))),
                tf.math.logical_and(
                    tf.greater_equal(v, dv), 
                    tf.less(v, dv*2)): lambda: ((tf.constant(0, tf.float32), tf.constant(1, tf.float32), 1 + (dv - v)/dv)),
                tf.math.logical_and(
                    tf.greater_equal(v, dv*2), 
                    tf.less(v, dv*3)): lambda: ((v - 2*dv)/dv, tf.constant(1, tf.float32), tf.constant(0, tf.float32))},
                default=lambda: (tf.constant(1, tf.float32), 1+(3*dv - v)/dv, tf.constant(0, tf.float32)),
                exclusive=True)

        a = tf.range(num)
        cr, cg, cb = tf.map_fn(lambda x: color_jet(x, num-1), a, dtype=(tf.float32, tf.float32, tf.float32))
        return tf.stack([cr, cg, cb], axis=1)

    def model_fn(self, features, labels, mode, params):
        cfg = self.cfg

        tpu = params['tpu']
        stride = params['stride']
        self.num_outputs = params['num_outputs']
        self.out_imsize = np.array([params['height'], params['width']])
        self.sm_size = np.ceil(self.out_imsize / (stride * 2)).astype(np.int32) * 2

        heads = self.get_net(features)
        X,Y,P = self.extract_predictions(heads)
        #batch x num_joints x num_outputs
        predictions = {
            'x': X,
            'y': Y,
            'p': P,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            if tpu:
                return TF.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)
            else:
                return TF.estimator.EstimatorSpec(mode, predictions=predictions)

        batch = tf.shape(features)[0]
        height = tf.shape(features)[1]
        width = tf.shape(features)[2]

        if not tpu:
            with tf.name_scope('summary') as scope:
                n = cfg.num_joints*self.num_outputs
                colormap = self.make_colormap(n)
                TF.summary.image('input', features)

                pt_y = tf.linspace(float(0), tf.cast(height-1, tf.float32),  tf.cast(height, tf.int32)) + .5
                pt_y = tf.reshape(pt_y, [-1, 1])
                pt_y = tf.tile(pt_y, [1, width])

                pt_x = tf.linspace(float(0), tf.cast(width-1, tf.float32), tf.cast(width, tf.int32)) + .5
                pt_x = tf.reshape(pt_x, [1, -1])
                pt_x = tf.tile(pt_x, [height, 1])

                dxp = tf.tile(tf.reshape(predictions['x'],[batch,1,1,-1]),[1,height,width,1])
                dxp = dxp - tf.tile(tf.reshape(pt_x, [1,height,width,1]),[batch,1,1,n])
                dyp = tf.tile(tf.reshape(predictions['y'],[batch,1,1,-1]),[1,height,width,1]) 
                dyp = dyp - tf.tile(tf.reshape(pt_y, [1,height,width,1]),[batch,1,1,n])                            
                distp = dxp ** 2 + dyp ** 2
                maskp = tf.argmax(tf.cast(distp <= 12, tf.int32), axis=3)
                maskp = tf.where_v2(
                        tf.tile(tf.reduce_any(distp <= 12, axis=3, keepdims=True),[1,1,1,3]),
                        tf.gather(colormap, maskp), 0)
                maskp = tf.clip_by_value(maskp, 0, 1)
                
                show_pred = features / 255. * 0.25 + maskp * 0.75
                TF.summary.image('prediction', show_pred)

                if 'bodypart' in labels:
                    bodyparts = labels['bodypart']
                    l_x,l_y = tf.unstack(tf.reshape(bodyparts, [batch, -1, 2]), axis=2)

                    dxl = tf.tile(tf.reshape(l_x,[batch,1,1,-1]),[1,height,width,1])
                    dxl = dxl - tf.tile(tf.reshape(pt_x, [1,height,width,1]),[batch,1,1,n])
                    dyl = tf.tile(tf.reshape(l_y,[batch,1,1,-1]),[1,height,width,1]) 
                    dyl = dyl - tf.tile(tf.reshape(pt_y, [1,height,width,1]),[batch,1,1,n])                            
                    distl = dxl ** 2 + dyl ** 2
                    maskl = tf.argmax(tf.cast(distl <= 12, tf.int32), axis=3)
                    cond1 = tf.tile(tf.reduce_any(distl <= 12, axis=3, keepdims=True),[1,1,1,3])
                    cond2 = tf.tile(tf.reshape(tf.reduce_any(tf.logical_not(tf.logical_or(tf.math.is_nan(l_x),tf.math.is_nan(l_y))),axis=1),[-1,1,1,1]),[1,height,width,3])
                    maskl = tf.where_v2(
                        tf.logical_and(cond1, cond2),
                        tf.gather(colormap, maskl), 0)
                    maskl = tf.clip_by_value(maskl, 0, 1)
                
                    show_label = features / 255. * 0.25 + maskl * 0.75
                    TF.summary.image('label', show_label)

        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = labels['part_score_weights'] if  weigh_part_predictions else 1.0

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
            with tf.name_scope('losses') as scope:
                for k, t in loss.items():
                    TF.summary.scalar(k, t)

        bodyparts = labels['bodypart']
        l_x,l_y = tf.unstack(tf.reshape(bodyparts, [batch, -1, 2]), axis=2)
        l_x = tf.reshape(l_x, [batch, -1, self.num_outputs])
        l_y = tf.reshape(l_y, [batch, -1, self.num_outputs])

        def calc_pixel_error(l_x, l_y, p_x, p_y, p_p, th_p):
            d = (l_x - p_x) ** 2 + (l_y - p_y) ** 2
            d = tf.where_v2(tf.math.greater(d, 0), tf.math.sqrt(d), 0.)
            label_exists = tf.logical_not(tf.logical_or(tf.math.is_nan(l_x), tf.math.is_nan(l_y)))
            return tf.boolean_mask(d, tf.logical_and(label_exists, tf.math.greater(p_p, th_p)))

        def metric_fn(l_x, l_y, p_x, p_y, p_p):
            return {
                    'pixel_error_0.1': TF.metrics.mean(calc_pixel_error(l_x, l_y, p_x, p_y, p_p, 0.1)),
                    'pixel_error_0.5': TF.metrics.mean(calc_pixel_error(l_x, l_y, p_x, p_y, p_p, 0.5)),
                    'pixel_error_0.9': TF.metrics.mean(calc_pixel_error(l_x, l_y, p_x, p_y, p_p, 0.9)),
                    }

        metrics = metric_fn(l_x, l_y, X, Y, P)
        if not tpu:
            with tf.name_scope('pixel_error') as scope:
                TF.summary.scalar('pixel_error_0.1', tf.math.reduce_mean(calc_pixel_error(l_x, l_y, X, Y, P, 0.1)))
                TF.summary.scalar('pixel_error_0.5', tf.math.reduce_mean(calc_pixel_error(l_x, l_y, X, Y, P, 0.5)))
                TF.summary.scalar('pixel_error_0.9', tf.math.reduce_mean(calc_pixel_error(l_x, l_y, X, Y, P, 0.9)))

        if mode == tf.estimator.ModeKeys.EVAL:    
            if tpu:
                return TF.estimator.tpu.TPUEstimatorSpec(
                        mode=mode, loss=total_loss, eval_metrics=(metric_fn, [l_x, l_y, X, Y, P]))

            else:
                return TF.estimator.EstimatorSpec(
                        mode=mode,
                        loss=total_loss,
                        eval_metric_ops=metrics)


        with tf.name_scope('optimizer') as scope:
            global_step = TF.train.get_or_create_global_step()
 
            vals = [s[0] for s in cfg.multi_step]
            if tpu:
                boundaries = [s[1] // 4 for s in cfg.multi_step]
            else:
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

        if tpu:
            return TF.estimator.tpu.TPUEstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op)
        else:
            return TF.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)
