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
vers = (tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF=tf.compat.v1
else:
    TF=tf

class PoseDataset:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_training_dataset(self):
        cfg = self.cfg
        self.width = cfg['train_width']
        self.height = cfg['train_height']
        datafile = cfg['train_set']
        return (tf.data.TFRecordDataset(datafile)
                .map(self.read_tfrecord, num_parallel_calls=8)
                .map(self.random_brightness, num_parallel_calls=8)
                .map(self.distorted_image, num_parallel_calls=8)
                .map(self.convert_jointdata, num_parallel_calls=8))

    def get_image_size(self, dataset_name='test'):
        if dataset_name == 'test':
            datafile = self.cfg['test_set']
        elif dataset_name == 'train':
            datafile = self.cfg['train_set']
        else:
            raise ValueError('dataset must be "train" or "test".')
        width = -1
        height = -1
        count = 0
        dataset = tf.data.TFRecordDataset(datafile).map(self.read_tfrecord_size, num_parallel_calls=15)
        with TF.Session() as sess:
            data = TF.data.make_one_shot_iterator(dataset).get_next()
            while True:
                try:
                    w, h = sess.run(data)
                except tf.errors.OutOfRangeError:
                    break
                count += 1
                if w > width:
                    width = w
                if h > height:
                    height = h
        return width, height, count

    def load_eval_dataset(self, width, height, dataset='test'):
        if dataset == 'test':
            datafile = self.cfg['test_set']
        elif dataset == 'train':
            datafile = self.cfg['train_set']
        else:
            raise ValueError('dataset must be "train" or "test".')
        cfg = self.cfg
        self.width = width
        self.height = height
        return tf.data.TFRecordDataset(datafile).map(self.read_tfrecord, num_parallel_calls=15).map(self.convert_evaldata)
    
    def load_predict_dataset(self, width, height, dataset='test'):
        if dataset == 'test':
            datafile = self.cfg['test_set']
        elif dataset == 'train':
            datafile = self.cfg['train_set']
        else:
            raise ValueError('dataset must be "train" or "test".')
        cfg = self.cfg
        self.width = width
        self.height = height
        return tf.data.TFRecordDataset(datafile).map(self.read_tfrecord, num_parallel_calls=15).map(self.convert_predictdata)
    
    def load_plot_dataset(self, width, height, dataset='test'):
        if dataset == 'test':
            datafile = self.cfg['test_set']
        elif dataset == 'train':
            datafile = self.cfg['train_set']
        else:
            raise ValueError('dataset must be "train" or "test".')
        cfg = self.cfg
        self.width = width
        self.height = height
        return tf.data.TFRecordDataset(datafile).map(self.read_tfrecord_filename).map(self.convert_plotdata)
    
    def read_tfrecord_size(self, serialized):
        with tf.name_scope('loadimages') as scope:
            cfg = self.cfg
            bodypart_num = cfg['num_joints']
            train_base = cfg['train_base_path']
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'filepath': tf.io.FixedLenFeature([], tf.string),
                    'bodypart': tf.io.FixedLenFeature([bodypart_num*2], tf.float32),
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'im_raw': tf.io.FixedLenFeature([], tf.string),
                })
            return features['width'], features['height']

    def read_tfrecord(self, serialized):
        with tf.name_scope('loadimages') as scope:
            cfg = self.cfg
            bodypart_num = cfg['num_joints']
            train_base = cfg['train_base_path']
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'filepath': tf.io.FixedLenFeature([], tf.string),
                    'bodypart': tf.io.FixedLenFeature([bodypart_num*2], tf.float32),
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'im_raw': tf.io.FixedLenFeature([], tf.string),
                })
            bodypart = tf.reshape(features['bodypart'], [-1,2])
            image = tf.io.decode_png(features['im_raw'], channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32) * 255.
            return image, bodypart

    def read_tfrecord_filename(self, serialized):
        with tf.name_scope('loadimages') as scope:
            cfg = self.cfg
            bodypart_num = cfg['num_joints']
            train_base = cfg['train_base_path']
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'filepath': tf.io.FixedLenFeature([], tf.string),
                    'bodypart': tf.io.FixedLenFeature([bodypart_num*2], tf.float32),
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'im_raw': tf.io.FixedLenFeature([], tf.string),
                })
            bodypart = tf.reshape(features['bodypart'], [-1,2])
            image = tf.io.decode_png(features['im_raw'], channels=3)
            return image, bodypart, features['filepath']

    def convert_predictdata(self, image, bodypart):
        with tf.name_scope('convert_jointdata') as scope:
            cfg = self.cfg
            stride = cfg.stride
            out_imsize = np.array([self.height, self.width])
            sm_size = np.ceil(out_imsize / (stride * 2)).astype(np.int32) * 2

            org_imsize = tf.shape(image)
            org_ratio = tf.cast(org_imsize[0], tf.float32)/tf.cast(org_imsize[1], tf.float32)
            out_ratio = tf.constant(self.height/self.width, tf.float32)

            scale = tf.case([(tf.less(out_ratio, org_ratio), lambda: tf.cast(org_imsize[1], tf.float32) / self.width)],
                    default=lambda: tf.cast(org_imsize[0], tf.float32) / self.height)
            pad = tf.case({tf.less(out_ratio, org_ratio): lambda: tf.stack([
                        (tf.constant(self.height, tf.float32) - tf.cast(org_imsize[0], tf.float32) * scale)/2,
                        tf.constant(0, tf.float32)]) },
                    default=lambda: tf.stack([
                        tf.constant(0, tf.float32), 
                        (tf.constant(self.width, tf.float32) - tf.cast(org_imsize[1], tf.float32) * scale)/2]),
                    exclusive=True)
            bodypart /= scale
            bodypart += pad
            image = TF.image.resize_image_with_pad(image, self.height, self.width)
            
            return image, { 'bodypart': bodypart }

    def convert_plotdata(self, image, bodypart, filepath):
        with tf.name_scope('convert_jointdata') as scope:
            cfg = self.cfg
            stride = cfg.stride
            out_imsize = np.array([self.height, self.width])
            sm_size = np.ceil(out_imsize / (stride * 2)).astype(np.int32) * 2

            org_imsize = tf.shape(image)
            org_ratio = tf.cast(org_imsize[0], tf.float32)/tf.cast(org_imsize[1], tf.float32)
            out_ratio = tf.constant(self.height/self.width, tf.float32)

            scale = tf.case([(tf.less(out_ratio, org_ratio), lambda: tf.cast(org_imsize[1], tf.float32) / self.width)],
                    default=lambda: tf.cast(org_imsize[0], tf.float32) / self.height)
            pad = tf.case({tf.less(out_ratio, org_ratio): lambda: tf.stack([
                        (tf.constant(self.height, tf.float32) - tf.cast(org_imsize[0], tf.float32) * scale)/2,
                        tf.constant(0, tf.float32)]) },
                    default=lambda: tf.stack([
                        tf.constant(0, tf.float32), 
                        (tf.constant(self.width, tf.float32) - tf.cast(org_imsize[1], tf.float32) * scale)/2]),
                    exclusive=True)
            bodypart /= scale
            bodypart += pad
            image = TF.image.resize_image_with_pad(image, self.height, self.width)
            
            return image, filepath, bodypart

    def convert_evaldata(self, image, bodypart):
        with tf.name_scope('convert_jointdata') as scope:
            cfg = self.cfg
            stride = cfg.stride
            out_imsize = np.array([self.height, self.width])
            sm_size = np.ceil(out_imsize / (stride * 2)).astype(np.int32) * 2

            org_imsize = tf.shape(image)
            org_ratio = tf.cast(org_imsize[0], tf.float32)/tf.cast(org_imsize[1], tf.float32)
            out_ratio = tf.constant(self.height/self.width, tf.float32)

            scale = tf.case([(tf.less(out_ratio, org_ratio), lambda: tf.cast(org_imsize[1], tf.float32) / self.width)],
                    default=lambda: tf.cast(org_imsize[0], tf.float32) / self.height)
            pad = tf.case({tf.less(out_ratio, org_ratio): lambda: tf.stack([
                        (tf.constant(self.height, tf.float32) - tf.cast(org_imsize[0], tf.float32) * scale)/2,
                        tf.constant(0, tf.float32)]) },
                    default=lambda: tf.stack([
                        tf.constant(0, tf.float32), 
                        (tf.constant(self.width, tf.float32) - tf.cast(org_imsize[1], tf.float32) * scale)/2]),
                    exclusive=True)
            bodypart /= scale
            bodypart += pad
            image = TF.image.resize_image_with_pad(image, self.height, self.width)
            
            results = tf.map_fn(lambda x: self.convert_jointdata_sub(x, sm_size, scale), bodypart)
            results = tf.transpose(results, [1, 2, 0, 3])
            scmap, locref_map_x, locref_map_y, weights = tf.unstack(results, axis=3)
            scmap_shape = tf.shape(scmap)
            locref_mask = tf.stack([scmap, scmap], axis=3)
            locref_mask = tf.reshape(locref_mask, [scmap_shape[0], scmap_shape[1], -1])
            locref_map = tf.stack([locref_map_x, locref_map_y], axis=3)
            locref_map = tf.reshape(locref_map, [scmap_shape[0], scmap_shape[1], -1])

            return image, { 'part_score_targets': scmap, 
                            'part_score_weights': weights, 
                            'locref_targets': locref_map, 
                            'locref_mask': locref_mask,
                            'bodypart': bodypart}


    def convert_jointdata(self, image, bodypart, scale):
        with tf.name_scope('convert_jointdata') as scope:
            cfg = self.cfg
            stride = cfg.stride
            out_imsize = np.array([self.height, self.width])
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
                            'locref_mask': locref_mask,
                            'bodypart': bodypart}

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

    def random_rotate_image(self, image, bodypart):
        with tf.name_scope('random_rotate') as scope:
            angle = tf.squeeze(tf.random.normal([1])) * 45 / 180 * np.pi

            expand = 1
            im_shape = tf.shape(image)
            height = im_shape[0]
            width = im_shape[1]
            channel = im_shape[2]

            x = tf.linspace(0., tf.cast(width*expand-1, tf.float32), width*expand)
            x = tf.tile(tf.expand_dims(x, axis=0), [height*expand, 1])

            y = tf.linspace(0., tf.cast(height*expand-1, tf.float32), height*expand)
            y = tf.tile(tf.expand_dims(y, axis=1), [1, width*expand])

            points = tf.stack([x, y], axis=2)

            def rotate(points, center, theta):
                rotation_matrix = tf.stack([tf.cos(theta),
                                          -tf.sin(theta),  
                                           tf.sin(theta),
                                           tf.cos(theta)])
                rotation_matrix = tf.reshape(rotation_matrix, (2,2))
                return tf.matmul(points - center, rotation_matrix) + center

            points = rotate(points, [width*expand/2, height*expand/2], angle)
            bodypart = rotate(bodypart, [width/2, height/2], -angle)

            x,y = tf.unstack(points, axis=2)
            indexes = tf.where_v2(tf.logical_and(
                tf.logical_and(tf.greater_equal(x, 0), tf.less(tf.cast(x,tf.int32), width*expand)),
                tf.logical_and(tf.greater_equal(y, 0), tf.less(tf.cast(y,tf.int32), height*expand))
            ), width*expand*tf.cast(y, tf.int32) + tf.cast(x, tf.int32), -1)

            if expand != 1:
                imex = tf.image.resize_images(image, [height*expand, width*expand])
            else:
                imex = image
            imex = tf.reshape(imex, [-1, channel])

            rotim = tf.where_v2(tf.tile(tf.expand_dims(tf.greater_equal(indexes, 0), axis=2), [1,1,channel]), 
                    tf.gather(imex, tf.clip_by_value(indexes, 0, (height*expand-1)*(width*expand-1))), 127.)

            if expand != 1:
                rotim = tf.image.resize_images(rotim, [height, width])

            return self.main_crop(rotim, bodypart)

    def random_brightness(self, image, bodypart):
        image = tf.image.random_brightness(image, 63.)
        image = tf.image.random_contrast(image, lower=0.25, upper=1.75)
        image = tf.clip_by_value(image, 0., 255.)
        return image, bodypart

    def distorted_image(self, image, bodypart):
        cfg = self.cfg
        return tf.cond(tf.squeeze(tf.random.uniform([1])) < cfg.random_rotation,
                true_fn=lambda: self.random_rotate_image(image, bodypart),
                false_fn=lambda: self.main_crop(image, bodypart))

    def main_crop(self, image, bodypart):
        with tf.name_scope('random_crop') as scope:
            cfg = self.cfg
            bodypart_num = cfg['num_joints']
            im_shape = tf.shape(image)
            image_shape = tf.cast(tf.shape(image), tf.float32)
            samples = tf.squeeze(tf.random.categorical(tf.math.log([[10. for _ in range(bodypart_num)]]), 1))
            select_joint = bodypart[samples, :]
            out_imsize = [self.height, self.width]

            def random_crop_fn(image, bodypart):
                begin = tf.concat([
                    tf.cond(im_shape[0] - out_imsize[0] > 0, 
                        true_fn=lambda:tf.random.uniform([1], minval=0, maxval=im_shape[0] - out_imsize[0],
                            dtype=tf.int32),
                        false_fn=lambda:tf.constant([0])),
                    tf.cond(im_shape[1] - out_imsize[1] > 0,
                        true_fn=lambda:tf.random.uniform([1], minval=0, maxval=im_shape[1] - out_imsize[1],
                            dtype=tf.int32),
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
                image = tf.image.resize_with_crop_or_pad(image, out_imsize[0], out_imsize[1])
                bodypart = bodypart - [tf.cast(begin[1],tf.float32)-pad[1], tf.cast(begin[0], tf.float32)-pad[0]]
                scale = tf.clip_by_value(tf.cast(size[:2][::-1] / out_imsize[::-1], tf.float32), 0, 1)
                bodypart /= scale 
                return image, bodypart, (scale[0]+scale[1])/2

            def crop_around_joint_fn(image, bodypart, select_joint, bounding_joint, min_object_covered):
                bounding_boxes = [[[tf.clip_by_value((select_joint[1]-bounding_joint)/image_shape[0], 0., 1.), 
                                    tf.clip_by_value((select_joint[0]-bounding_joint)/image_shape[1], 0., 1.), 
                                    tf.clip_by_value((select_joint[1]+bounding_joint)/image_shape[0], 0., 1.), 
                                    tf.clip_by_value((select_joint[0]+bounding_joint)/image_shape[1], 0., 1.)]]] 
                begin, size, _ = tf.image.sample_distorted_bounding_box(
                    tf.shape(image), bounding_boxes,
                    min_object_covered=min_object_covered,
                    aspect_ratio_range=[9.0/10.0, 10.0/9.0])
                    
                def success_fn(image, bodypart, begin, size):
                  image = tf.slice(image, begin, size)
                  image = tf.image.resize(image, out_imsize, method=tf.image.ResizeMethod.BICUBIC)
                  bodypart = bodypart - [begin[1], begin[0]]
                  scale = tf.cast(size[:2][::-1] / out_imsize[::-1], tf.float32)
                  bodypart /= scale
                  return image, bodypart, (scale[0]+scale[1])/2

                return tf.cond(tf.reduce_all(tf.equal(size[:2],im_shape[:2])),
                             true_fn=lambda: random_crop_fn(image, bodypart),
                             false_fn=lambda: success_fn(image, bodypart, begin, size))

            bounding_joint = cfg.bounding_joint
            min_object_covered = cfg.min_object_covered
            return tf.cond(tf.reduce_any(tf.math.is_nan(select_joint)),
                    true_fn=lambda: random_crop_fn(image, bodypart),
                    false_fn=lambda: tf.cond(
                        tf.squeeze(tf.random.uniform([1])) < cfg.cropratio,
                        true_fn=lambda: crop_around_joint_fn(image, bodypart, select_joint, 
                            bounding_joint, min_object_covered),
                        false_fn=lambda: random_crop_fn(image, bodypart)))

