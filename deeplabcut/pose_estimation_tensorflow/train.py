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

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.factory import create as create_dataset
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net


def train(config_yaml,displayiters,saveiters,maxiters,max_to_keep=5):
    tpu = os.getenv('TPU', False)
    start_path=os.getcwd()
    os.chdir(str(Path(config_yaml).parents[0])) #switch to folder of config_yaml (for logging)
    TF.logging.set_verbosity(TF.logging.INFO)

    cfg = load_config(config_yaml)
    #cfg['batch_size']=1 #in case this was edited for analysis.

    project_path = cfg['project_path']
    trainbase_path = cfg['train_base_path']
    if project_path != trainbase_path:
        print('copy to training folder...')
        if not tf.io.gfile.exists(cfg['test_set']):
            tf.io.gfile.copy(cfg['test_set'].replace(trainbase_path, project_path), cfg['test_set'])
        if not tf.io.gfile.exists(cfg['train_set']):
            tf.io.gfile.copy(cfg['train_set'].replace(trainbase_path, project_path), cfg['train_set'])
        resultfrom_path = str(Path(config_yaml).parents[0])
        resultto_path = resultfrom_path.replace(project_path, trainbase_path)
        def cp(base):
            for copy_file in tf.io.gfile.glob(os.path.join(base, '*')):
                if tf.io.gfile.isdir(copy_file):
                    output_path = copy_file.replace(project_path, trainbase_path)
                    tf.io.gfile.makedirs(output_path)
                    cp(copy_file)
                else:
                    output_path = copy_file.replace(project_path, trainbase_path)
                    if not tf.io.gfile.exists(output_path):
                        tf.io.gfile.copy(copy_file, output_path)
        cp(resultfrom_path)


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
    warm_start_settings = TF.estimator.WarmStartSettings(ckpt_to_initialize_from=cfg.init_weights, 
            vars_to_warm_start='resnet.*')

    train_width = cfg['train_width']
    train_height = cfg['train_height']
    test_width, test_height, test_count = create_dataset(cfg).get_image_size('test')
    if tpu:
        if os.getenv('COLAB_TPU_ADDR', None):
            tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
            tpu_cluster_resolver = TF.distribute.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        else:
            tpu_cluster_resolver = TF.distribute.cluster_resolver.TPUClusterResolver()

        run_config = TF.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                session_config=TF.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True),
                    tpu_config=TF.estimator.tpu.TPUConfig(
                        iterations_per_loop='2m',
                    ),
                save_checkpoints_steps=None,
                save_checkpoints_secs=5*60,
                keep_checkpoint_max=max_to_keep)

        dlc = TF.estimator.tpu.TPUEstimator(
                model_fn=model_fn,
                model_dir=trainpath,
                use_tpu=True,
                train_batch_size=cfg.batch_size*8,
                eval_batch_size=cfg.batch_size*8,
                predict_batch_size=cfg.batch_size*8,
                params={
                    'tpu': tpu,
                    'width': train_width,
                    'height': train_height,
                    'stride': cfg.stride,
                    'num_outputs': 1,
                    },
                config=run_config,
                warm_start_from=warm_start_settings)
        dlc_test = TF.estimator.tpu.TPUEstimator(
                model_fn=model_fn,
                model_dir=trainpath,
                use_tpu=True,
                train_batch_size=cfg.batch_size*8,
                eval_batch_size=cfg.batch_size*8,
                predict_batch_size=cfg.batch_size*8,
                params={
                    'tpu': tpu,
                    'width': test_width,
                    'height': test_height,
                    'stride': cfg.stride,
                    'num_outputs': 1,
                    },
                config=run_config)
    else:
        run_config = TF.estimator.RunConfig(
                save_checkpoints_steps=None,
                save_checkpoints_secs=5*60,
                keep_checkpoint_max=max_to_keep)

        dlc = TF.estimator.Estimator(
                model_fn=model_fn,
                model_dir=trainpath,
                params={
                    'tpu': tpu,
                    'width': train_width,
                    'height': train_height,
                    'stride': cfg.stride,
                    'num_outputs': 1,
                    },
                config=run_config,
                warm_start_from=warm_start_settings)
        dlc_test = TF.estimator.Estimator(
                model_fn=model_fn,
                model_dir=trainpath,
                params={
                    'tpu': tpu,
                    'width': test_width,
                    'height': test_height,
                    'stride': cfg.stride,
                    'num_outputs': 1,
                    },
                config=run_config)

    print("Training parameter:")
    print(cfg)
    print("Starting training....")

    def train_input_fn(batch_size=1):
        print('batch_size', batch_size)
        dataset = create_dataset(cfg).load_training_dataset()
        dataset = dataset.shuffle(1000).repeat()
        return dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size)

    def eval_input_fn(batch_size=1, width=-1, height=-1):
        print('batch_size', batch_size)
        dataset = create_dataset(cfg).load_eval_dataset(width, height, 'test')
        dataset = dataset.shuffle(1000).repeat()
        return dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size)

    eval_iter = 50000
    lastckpt = dlc.latest_checkpoint()
    if lastckpt:
        last_num = (int(lastckpt.split('-')[-1]) - 1) // eval_iter * eval_iter
        if last_num < 0:
            last_num = 0
    else:
        last_num = 0

    try:
        if tpu:
            max_iter = max_iter
            r = list(range(last_num+eval_iter,max_iter,eval_iter))
            r += [max_iter]
            for i in sorted(r):
                dlc.train(input_fn=lambda params: train_input_fn(params["batch_size"]), max_steps=i)
                result = dlc_test.evaluate(
                    input_fn=lambda params: eval_input_fn(params["batch_size"], params['width'], params['height']), 
                    steps=test_count//8+1)
                print(result)
        else:
            r = list(range(last_num+eval_iter,max_iter,eval_iter))
            r += [max_iter]
            for i in sorted(r):
                dlc.train(input_fn=lambda:train_input_fn(cfg.batch_size), max_steps=i)
                result = dlc_test.evaluate(
                    input_fn=lambda: eval_input_fn(cfg.batch_size, test_width, test_height), 
                    steps=test_count)
                print(result)
    except KeyboardInterrupt:
        pass
 
    if project_path != trainbase_path:
        print('copy back training results...')
        resultto_path = str(Path(config_yaml).parents[0])
        resultfrom_path = resultto_path.replace(project_path, trainbase_path)
        def cp(base):
            for copy_file in tf.io.gfile.glob(os.path.join(base, '*')):
                if tf.io.gfile.isdir(copy_file):
                    output_path = copy_file.replace(trainbase_path, project_path)
                    tf.io.gfile.makedirs(output_path)
                    cp(copy_file)
                else:
                    output_path = copy_file.replace(trainbase_path, project_path)
                    tf.io.gfile.copy(copy_file, output_path, overwrite=True)
        cp(resultfrom_path)

    #return to original path.
    os.chdir(str(start_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml configuration file.')
    cli_args = parser.parse_args()

    train(Path(cli_args.config).resolve())
