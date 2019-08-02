"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

"""
import logging, os
import threading
import argparse
from pathlib import Path
import tensorflow as tf
vers = (tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF=tf.compat.v1
else:
    TF=tf
import tensorflow.contrib.slim as slim
import numpy as np

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.factory import create as create_dataset
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import get_batch_spec
from deeplabcut.pose_estimation_tensorflow.util.logging import setup_logging
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch

class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr

def setup_preloading(batch_spec):
    placeholders = {name: TF.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20
    vers = (tf.__version__).split('.')
    if int(vers[0])==1 and int(vers[1])>12:
        q = tf.queue.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    else:
        q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = TF.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t

def get_optimizer(loss_op, cfg):
    learning_rate = TF.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = TF.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = TF.train.AdamOptimizer(cfg.adam_lr)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op

def train(config_yaml,displayiters,saveiters,maxiters,max_to_keep=5):
    tpu = os.getenv('TPU', False)
    start_path=os.getcwd()
    os.chdir(str(Path(config_yaml).parents[0])) #switch to folder of config_yaml (for logging)
    TF.logging.set_verbosity(TF.logging.INFO)
    setup_logging()

    cfg = load_config(config_yaml)
    #cfg['batch_size']=1 #in case this was edited for analysis.

    if cfg.deterministic:
        tf.set_random_seed(42)

    if maxiters==None:
        max_iter = int(cfg.multi_step[-1][1])
    else:
        max_iter = min(int(cfg.multi_step[-1][1]),int(maxiters))
        #display_iters = max(1,int(displayiters))
        print("Max_iters overwritten as",max_iter)

    if displayiters==None:
        display_iters = max(1,int(cfg.display_iters))
    else:
        display_iters = max(1,int(displayiters))
        print("Display_iters overwritten as",display_iters)

    if saveiters==None:
        save_iters=max(1,int(cfg.save_iters))

    else:
        save_iters=max(1,int(saveiters))
        print("Save_iters overwritten as",save_iters)

    model_fn = lambda features, labels, mode, params: pose_net(cfg).model_fn(features, labels, mode, params)

    trainpath=str(config_yaml).split('pose_cfg.yaml')[0]
    trainpath=trainpath.replace(cfg['project_path'],cfg['train_base_path'])
    if len(tf.io.gfile.glob(os.path.join(trainpath,'*.index'))) > 0:
        warm_start_settings = TF.estimator.WarmStartSettings(ckpt_to_initialize_from=trainpath)
    else:
        warm_start_settings = TF.estimator.WarmStartSettings(ckpt_to_initialize_from=cfg.init_weights, 
            vars_to_warm_start='resnet.*')

    if tpu:
        if os.getenv('COLAB_TPU_ADDR', None):
            tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
            tpu_cluster_resolver = TF.distribute.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        else:
            tpu_cluster_resolver = TF.distribute.cluster_resolver.TPUClusterResolver()

    if tpu:
        run_config = TF.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                session_config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True),
                    tpu_config=TF.estimator.tpu.TPUConfig(
                        iterations_per_loop='5m',
                    ),
                save_checkpoints_steps=None,
                save_checkpoints_secs=5*60,
                keep_checkpoint_max=max_to_keep)

        dlc = tf.estimator.tpu.TPUEstimator(
                model_fn=model_fn,
                model_dir=trainpath,
                use_tpu=True,
                train_batch_size=cfg.batch_size*8,
                eval_batch_size=cfg.batch_size*8,
                predict_batch_size=cfg.batch_size*8,
                params={},
                config=run_config,
                warm_start_from=warm_start_settings)
    else:
        run_config = TF.estimator.RunConfig(
                save_checkpoints_steps=None,
                save_checkpoints_secs=5*60,
                keep_checkpoint_max=max_to_keep)

        dlc = tf.estimator.Estimator(
                model_fn=model_fn,
                model_dir=trainpath,
                params={},
                config=run_config,
                warm_start_from=warm_start_settings)

    #stats_path = Path(config_yaml).with_name('learning_stats.csv')
    #lrf = open(str(stats_path), 'w')

    print("Training parameter:")
    print(cfg)
    print("Starting training....")

    def train_input_fn(batch_size=1):
        print('batch_size', batch_size)
        dataset = create_dataset(cfg).load_dataset()
        return dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size*2)

    if tpu:
        dlc.train(input_fn=lambda params: train_input_fn(params["batch_size"]), max_steps=max_iter)
    else:
        dlc.train(input_fn=lambda:train_input_fn(cfg.batch_size), max_steps=max_iter)

    #lrf.close()
    #return to original path.
    os.chdir(str(start_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml configuration file.')
    cli_args = parser.parse_args()

    train(Path(cli_args.config).resolve())
