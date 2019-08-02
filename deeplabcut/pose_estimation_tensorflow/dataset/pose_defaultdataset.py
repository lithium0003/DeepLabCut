'''
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''


'''
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import os
import logging
import random as rand
import numpy as np
from numpy import array as arr
from numpy import concatenate as cat
import tensorflow as tf


class PoseDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        train_width = cfg['train_width']
        train_height = cfg['train_height']
        cfg.out_imsize = [train_height, train_width]

    def load_dataset(self, training=True):
        cfg = self.cfg
        if training:
            datafile = cfg['train_set']
        else:
            datafile = cfg['test_set']
        return tf.data.TFRecordDataset(datafile).repeat().shuffle(1024).map(self.read_tfrecord, num_parallel_calls=5).map(self.distorted_image, num_parallel_calls=5).map(self.convert_jointdata, num_parallel_calls=5)

    def read_tfrecord(self, serialized):
        cfg = self.cfg
        bodypart_num = cfg['num_joints']
        train_base = cfg['train_base_path']
        features = tf.io.parse_single_example(
            serialized,
            features={
                'filepath': tf.io.FixedLenFeature([], tf.string),
                'bodypart': tf.io.FixedLenFeature([bodypart_num*2], tf.float32),
            })
        filepath = tf.strings.join([train_base, features['filepath']], separator='/')
        bodypart = tf.reshape(features['bodypart'], [-1,2])
        image = tf.io.decode_png(tf.io.read_file(filepath), channels=3)
        return image, bodypart

    def convert_jointdata(self, image, bodypart, scale):
        cfg = self.cfg
        stride = cfg.stride
        out_imsize = np.array(cfg.out_imsize)
        sm_size = np.ceil(out_imsize / (stride * 2)).astype(np.int32) * 2

        results = tf.map_fn(lambda x: self.convert_jointdata_sub(x, sm_size, scale), bodypart)
        results = tf.transpose(results, [1, 2, 0, 3])
        scmap, locref_map_x, locref_map_y, weights = tf.unstack(results, axis=3)
        scmap_shape = tf.shape(scmap)
        locref_mask = tf.stack([scmap, scmap], axis=3)
        locref_mask = tf.reshape(locref_mask, [scmap_shape[0], scmap_shape[1], -1])
        locref_map = tf.stack([locref_map_x, locref_map_y], axis=3)
        locref_map = tf.reshape(locref_map, [scmap_shape[0], scmap_shape[1], -1])

        image.set_shape([out_imsize[0], out_imsize[1], 3])

        return image, { 'part_score_targets': scmap, 
                        'part_score_weights': weights, 
                        'locref_targets': locref_map, 
                        'locref_mask': locref_mask}

    def convert_jointdata_sub(self, joints, shape, scale):
        pos_dist_thresh = self.cfg.pos_dist_thresh
        locref_stdev = self.cfg.locref_stdev
        stride = self.cfg.stride 

        dist_thresh = pos_dist_thresh / scale
        dist_thresh_sq = dist_thresh ** 2
        locref_scale = 1.0 / locref_stdev

        j_x = joints[0]
        j_y = joints[1]

        width = shape[1]
        height = shape[0]
          
        z = tf.zeros([height, width], dtype=tf.float32)

        pt_y = tf.linspace(float(0), tf.cast(height-1, tf.float32),  tf.cast(height, tf.int32)) * stride + stride/2
        pt_y = tf.reshape(pt_y, [-1, 1])
        pt_y = tf.tile(pt_y, [1, width])

        pt_x = tf.linspace(float(0), tf.cast(width-1, tf.float32), tf.cast(width, tf.int32)) * stride + stride/2
        pt_x = tf.reshape(pt_x, [1, -1])
        pt_x = tf.tile(pt_x, [height, 1])

        dx = j_x - pt_x
        dy = j_y - pt_y
        dist = dx ** 2 + dy ** 2

        scmap = tf.cast(dist <= dist_thresh_sq, tf.float32)
        locref_map_x = tf.cast(dist <= dist_thresh_sq, tf.float32) * dx * locref_scale
        locref_map_y = tf.cast(dist <= dist_thresh_sq, tf.float32) * dy * locref_scale
        weights = tf.ones(tf.shape(scmap))

        return tf.cond(tf.math.logical_or(tf.reduce_any(tf.math.is_nan(joints)), scale <= 0),
                true_fn=lambda :tf.stack([z, z, z, weights], axis=2),
                false_fn=lambda: tf.stack([scmap, locref_map_x, locref_map_y, weights], axis=2))

    def distorted_image(self, image, bodypart):
        cfg = self.cfg
        bodypart_num = cfg['num_joints']
        im_shape = tf.shape(image)
        image_shape = tf.cast(tf.shape(image), tf.float32)
        samples = tf.squeeze(tf.random.categorical(tf.math.log([[10. for _ in range(bodypart_num)]]), 1))
        select_joint = bodypart[samples, :]
        out_imsize = cfg.out_imsize

        def random_crop_fn(image, bodypart):
            begin = tf.concat([
                tf.cond(im_shape[0] - out_imsize[0] > 0, 
                    true_fn=lambda:tf.random.uniform([1], minval=0, maxval=im_shape[0] - out_imsize[0], dtype=tf.int32), 
                    false_fn=lambda:tf.constant([0])),
                tf.cond(im_shape[1] - out_imsize[1] > 0, 
                    true_fn=lambda:tf.random.uniform([1], minval=0, maxval=im_shape[1] - out_imsize[1], dtype=tf.int32), 
                    false_fn=lambda:tf.constant([0])),
                tf.constant([0])], 0)
            size = tf.stack([tf.clip_by_value(out_imsize[0], 0, im_shape[0]-begin[0]),
                            tf.clip_by_value(out_imsize[1], 0, im_shape[1]-begin[1]),
                            tf.constant(-1)])
            pad = tf.stack([
                tf.cond(im_shape[0] - out_imsize[0] > 0, 
                    true_fn=lambda:tf.constant(0.), 
                    false_fn=lambda:tf.cast((out_imsize[0]-im_shape[0])/2, tf.float32)),
                tf.cond(im_shape[1] - out_imsize[1] > 0, 
                    true_fn=lambda:tf.constant(0.), 
                    false_fn=lambda:tf.cast((out_imsize[1]-im_shape[1])/2, tf.float32))])
            image = tf.slice(image, begin, size)
            image = tf.image.convert_image_dtype(image, tf.float32) * 255.
            image = tf.image.resize_with_crop_or_pad(image, out_imsize[0], out_imsize[1])
            bodypart = bodypart - [tf.cast(begin[1],tf.float32)-pad[1], tf.cast(begin[0], tf.float32)-pad[0]]
            scale = tf.clip_by_value(tf.cast(size[:2][::-1] / out_imsize[::-1], tf.float32), 0, 1)
            bodypart /= scale 
            return image, bodypart, (scale[0]+scale[1])/2

        def crop_around_joint_fn(image, bodypart, select_joint):
            bounding_boxes = [[[tf.clip_by_value((select_joint[1] - 100)/image_shape[0], 0., 1.), 
                                  tf.clip_by_value((select_joint[0] - 100)/image_shape[1], 0., 1.), 
                                  tf.clip_by_value((select_joint[1] + 100)/image_shape[0], 0., 1.), 
                                  tf.clip_by_value((select_joint[0] + 100)/image_shape[1], 0., 1.)]]] 
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(image), bounding_boxes,
                min_object_covered=1.0,
                aspect_ratio_range=[1.0, 1.0])
                
            def success_fn(image, bodypart, begin, size):
              image = tf.slice(image, begin, size)
              image = tf.image.convert_image_dtype(image, tf.float32) * 255.
              image = tf.image.resize(image, out_imsize, method=tf.image.ResizeMethod.BICUBIC)
              bodypart = bodypart - [begin[1], begin[0]]
              scale = tf.cast(size[:2][::-1] / out_imsize[::-1], tf.float32)
              bodypart /= scale
              return image, bodypart, (scale[0]+scale[1])/2

            return tf.cond(tf.reduce_all(tf.equal(size[:2],im_shape[:2])),
                         true_fn=lambda: random_crop_fn(image, bodypart),
                         false_fn=lambda: success_fn(image, bodypart, begin, size))

        return tf.cond(tf.reduce_any(tf.math.is_nan(select_joint)),
                true_fn=lambda: random_crop_fn(image, bodypart),
                false_fn=lambda: crop_around_joint_fn(image, bodypart, select_joint))

