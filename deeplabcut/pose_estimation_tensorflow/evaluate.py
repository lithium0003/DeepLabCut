"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import argparse

from deeplabcut.pose_estimation_tensorflow.dataset.factory import create as create_dataset
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

# Dependencies for anaysis
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def evaluate_network(config,Shuffles=[1],plotting = None,show_errors = True,comparisonbodyparts="all",gputouse=None):
    """
    Evaluates the network based on the saved models at different stages of the training network.\n
    The evaluation results are stored in the .h5 and .csv file under the subdirectory 'evaluation_results'.
    Change the snapshotindex parameter in the config file to 'all' in order to evaluate all the saved models.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Shuffles: list, optional
        List of integers specifying the shuffle indices of the training dataset. The default is [1]

    plotting: bool, optional
        Plots the predictions on the train and test images. The default is ``False``; if provided it must be either ``True`` or ``False``

    show_errors: bool, optional
        Display train and test errors. The default is `True``

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    Examples
    --------
    If you do not want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml', shuffle=[1])
    --------

    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',shuffle=[1],True)
    """
    import os
    from skimage import io
    import skimage.color

    from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
    from deeplabcut.utils import auxiliaryfunctions, visualization
    import tensorflow as tf
    vers = (tf.__version__).split('.')
    if int(vers[0])==1 and int(vers[1])>12:
        TF=tf.compat.v1
    else:
        TF=tf

    tpu = os.getenv('TPU', False)
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #
    TF.logging.set_verbosity(TF.logging.WARN)
#    tf.logging.set_verbosity(tf.logging.WARN)

    start_path=os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)


    project_path = cfg['project_path']
    trainbase_path = cfg['train_base_path']

    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    cmpbp_idx = np.array([cfg['bodyparts'].index(i) for i in comparisonbodyparts])
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(project_path+"/evaluation-results/"))
    for shuffle in Shuffles:
        for trainFraction in cfg["TrainingFraction"]:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            #datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
            modelfolder=os.path.join(trainbase_path,str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
            orgmodelfolder=os.path.join(project_path,str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
            path_test_config = Path(orgmodelfolder) / 'test' / 'pose_cfg.yaml'
            # Load meta data
            #data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))

            try:
                print(str(path_test_config))
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

            if project_path != trainbase_path:
                print('copy to training folder...')
                if not tf.io.gfile.exists(dlc_cfg['test_set']):
                    tf.io.gfile.copy(dlc_cfg['test_set'].replace(trainbase_path, project_path), dlc_cfg['test_set'])
                if not tf.io.gfile.exists(dlc_cfg['train_set']):
                    tf.io.gfile.copy(dlc_cfg['train_set'].replace(trainbase_path, project_path), dlc_cfg['train_set'])
                resultfrom_path = os.path.join(str(orgmodelfolder), 'train')
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

            #change batch size, if it was edited during analysis!
            dlc_cfg['batch_size']=1 #in case this was edited for analysis.
            #Create folder structure to store results.
            trainpath = os.path.join(str(modelfolder), 'train')
            evaluationfolder=os.path.join(project_path,str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg)))
            auxiliaryfunctions.attempttomakefolder(evaluationfolder,recursive=True)
            #path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'

            model_fn = lambda features, labels, mode, params: pose_net(dlc_cfg).model_fn(features, labels, mode, params)
    
            w1, h1, test_count = create_dataset(dlc_cfg).get_image_size('test')
            w2, h2, train_count = create_dataset(dlc_cfg).get_image_size('train')
            im_width = max(w1, w2)
            im_height = max(h1, h2)
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
                        )
                dlc = TF.estimator.tpu.TPUEstimator(
                        model_fn=model_fn,
                        model_dir=trainpath,
                        use_tpu=True,
                        train_batch_size=dlc_cfg.batch_size*8,
                        eval_batch_size=dlc_cfg.batch_size*8,
                        predict_batch_size=dlc_cfg.batch_size*8,
                        params={
                            'tpu': tpu,
                            'width': im_width,
                            'height': im_height,
                            'stride': dlc_cfg.stride,
                            'num_outputs': 1,
                            },
                        config=run_config)
            else:
                dlc = TF.estimator.Estimator(
                        model_fn=model_fn,
                        model_dir=trainpath,
                        params={
                            'tpu': tpu,
                            'width': im_width,
                            'height': im_height,
                            'stride': dlc_cfg.stride,
                            'num_outputs': 1,
                            },
                        )

            lastckpt = dlc.latest_checkpoint()
            if lastckpt:
                ckpt_iter = int(lastckpt.split('-')[-1])
            else:
                ckpt_iter = 0

            def predict_train_input_fn(batch_size=1, width=-1, height=-1):
                print('batch_size', batch_size)
                dataset = create_dataset(dlc_cfg).load_predict_dataset(width, height, 'train')
                return dataset.batch(batch_size).prefetch(batch_size)

            def predict_test_input_fn(batch_size=1, width=-1, height=-1):
                print('batch_size', batch_size)
                dataset = create_dataset(dlc_cfg).load_predict_dataset(width, height, 'test')
                return dataset.batch(batch_size).prefetch(batch_size)

            final_result = []
            try:
                train_predicted = np.zeros([train_count, 3*len(dlc_cfg['all_joints_names'])])
                test_predicted = np.zeros([train_count, 3*len(dlc_cfg['all_joints_names'])])
                if tpu:
                    train_predictions = dlc.predict(
                        input_fn=lambda params: predict_train_input_fn(params["batch_size"], params['width'], params['height']))

                    image_count = 0
                    for predict in train_predictions:
                        train_predicted[image_count, 0::3] = predict['x'].flatten()
                        train_predicted[image_count, 1::3] = predict['y'].flatten()
                        train_predicted[image_count, 2::3] = predict['p'].flatten()
                        image_count += 1
                    
                    test_predictions = dlc.predict(
                        input_fn=lambda params: predict_test_input_fn(params["batch_size"], params['width'], params['height']))

                    image_count = 0
                    for predict in test_predictions:
                        test_predicted[image_count, 0::3] = predict['x'].flatten()
                        test_predicted[image_count, 1::3] = predict['y'].flatten()
                        test_predicted[image_count, 2::3] = predict['p'].flatten()
                        image_count += 1
                else:
                    train_predictions = dlc.predict(
                        input_fn=lambda: predict_train_input_fn(dlc_cfg.batch_size, im_width, im_height))

                    image_count = 0
                    for predict in train_predictions:
                        train_predicted[image_count, 0::3] = predict['x'].flatten()
                        train_predicted[image_count, 1::3] = predict['y'].flatten()
                        train_predicted[image_count, 2::3] = predict['p'].flatten()
                        image_count += 1
                    
                    test_predictions = dlc.predict(
                        input_fn=lambda: predict_test_input_fn(dlc_cfg.batch_size, im_width, im_height))

                    image_count = 0
                    for predict in test_predictions:
                        test_predicted[image_count, 0::3] = predict['x'].flatten()
                        test_predicted[image_count, 1::3] = predict['y'].flatten()
                        test_predicted[image_count, 2::3] = predict['p'].flatten()
                        image_count += 1
                    
                if plotting == True:
                    print("Plotting...")
                    colors = visualization.get_cmap(len(comparisonbodyparts),name=cfg['colormap'])
                    foldername=os.path.join(str(evaluationfolder),'LabeledImages_' + str(ckpt_iter))
                    auxiliaryfunctions.attempttomakefolder(foldername)
                
                def process_data(predicted, dataset, im_count, isTrain):
                    with TF.Session() as sess:
                        count = 0
                        RMSE = np.zeros([im_count, len(dlc_cfg['all_joints_names'])])
                        RMSEpcutoff = np.zeros([im_count, len(dlc_cfg['all_joints_names'])])
                        data = TF.data.make_one_shot_iterator(dataset).get_next()
                        while True:
                            try:
                                im, name, bodypart = sess.run(data)
                                name = name.decode('utf-8')
                            except tf.errors.OutOfRangeError:
                                break

                            def calc_distance(labeled, predict):
                                mask = predict[:,2] < cfg["pcutoff"]
                                rmse = np.sqrt((labeled[:,0] - predict[:,0])**2 + (labeled[:,1] - predict[:,1])**2)
                                rmse_mask = rmse
                                rmse_mask[mask] = float('nan')
                                return rmse.flatten(), rmse_mask.flatten()
                            
                            pred = predicted[count, :].reshape((-1,3))
                            RMSE[count,:],RMSEpcutoff[count,:] = calc_distance(bodypart, pred)
                            if plotting == True:
                                visualization.PlottingandSaveLabeledFrame(name,im,bodypart,pred,isTrain,
                                        cfg,colors,cmpbp_idx,foldername)
                            count += 1
                        return np.nanmean(RMSE[:,cmpbp_idx].flatten()), np.nanmean(RMSEpcutoff[:,cmpbp_idx].flatten())

                train_dataset = create_dataset(dlc_cfg).load_plot_dataset(im_width, im_height, 'train')
                trainerror, trainerrorpcutoff = process_data(train_predicted, train_dataset, train_count, True)

                test_dataset = create_dataset(dlc_cfg).load_plot_dataset(im_width, im_height, 'test')
                testerror, testerrorpcutoff = process_data(test_predicted, test_dataset, test_count, False)
   
                results = [ckpt_iter,int(100 * trainFraction),shuffle,np.round(trainerror,2),np.round(testerror,2),cfg["pcutoff"],np.round(trainerrorpcutoff,2), np.round(testerrorpcutoff,2)]
                final_result.append(results)

                if show_errors == True:
                    print("Results for",ckpt_iter," training iterations:", int(100 * trainFraction), shuffle, "train error:",np.round(trainerror,2), "pixels. Test error:", np.round(testerror,2)," pixels.")
                    print("With pcutoff of", cfg["pcutoff"]," train error:",np.round(trainerrorpcutoff,2), "pixels. Test error:", np.round(testerrorpcutoff,2), "pixels")
                    print("Thereby, the errors are given by the average distances between the labels by DLC and the scorer.")

            except KeyboardInterrupt:
                pass
                    
            TF.reset_default_graph()
            
            #print(final_result)
            DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,ckpt_iter)
            make_results_file(final_result,evaluationfolder,DLCscorer)
            print("The network is evaluated and the results are stored in the subdirectory 'evaluation_results'.")
            print("If it generalizes well, choose the best model for prediction and update the config file with the appropriate index for the 'snapshotindex'.\nUse the function 'analyze_video' to make predictions on new videos.")
            print("Otherwise consider retraining the network (see DeepLabCut workflow Fig 2)")

    #returning to intial folder
    os.chdir(str(start_path))

def make_results_file(final_result,evaluationfolder,DLCscorer):
    """
    Makes result file in .h5 and csv format and saves under evaluation_results directory
    """
    col_names = ["Training iterations:","%Training dataset","Shuffle number"," Train error(px)"," Test error(px)","p-cutoff used","Train error with p-cutoff","Test error with p-cutoff"]
    df = pd.DataFrame(final_result, columns = col_names)
    df.to_hdf(os.path.join(str(evaluationfolder),DLCscorer + '-results' + '.h5'),'df_with_missing',format='table',mode='w')
    df.to_csv(os.path.join(str(evaluationfolder),DLCscorer + '-results' + '.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
