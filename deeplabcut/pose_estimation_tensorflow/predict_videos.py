"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

####################################################
# Dependencies
####################################################

import os.path
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
vers = (tf.__version__).split('.')
if int(vers[0])==1 and int(vers[1])>12:
    TF=tf.compat.v1
else:
    TF=tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
from skimage.util import img_as_ubyte
from PIL import Image
import io

def convert_analyze_videos(config,videos,dest,videotype='avi'):
    start_path=os.getcwd() #record cwd to return to this directory in the end
    cfg = auxiliaryfunctions.read_config(config)
    
    Videos=auxiliaryfunctions.Getlistofvideos(videos,videotype)

    if len(Videos)>0:
        #looping over videos
        for video in Videos:
            ConvertVideo(video,dest)

        os.chdir(str(start_path))

def ConvertVideo(video, dest):
    import threading
    import multiprocessing
    from multiprocessing import Queue
    import collections
    import concurrent

    cap=cv2.VideoCapture(video)
    videoname = Path(video).stem
    if tf.io.gfile.isdir(os.path.join(dest, videoname)):
        print(os.path.join(dest, videoname))
        print('converted directory exists, aborted.')
        return

    tf.io.gfile.makedirs(os.path.join(dest, videoname))

    fps = cap.get(5) #https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    nframes = int(cap.get(7))
    duration=nframes*1./fps
    size=(int(cap.get(4)),int(cap.get(3)))

    ny,nx=size
    print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
    print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)

    num_worker = max(1, multiprocessing.cpu_count()-2)
    
    def process_image(index, image):

        def make_example(im_raw, width, height, i, nframes):        
            return tf.train.Example(features=tf.train.Features(feature={      
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'frame_no': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'nframes': tf.train.Feature(int64_list=tf.train.Int64List(value=[nframes])),
                'im_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw])),
                }))

        img_pil = Image.fromarray(image)
        with io.BytesIO() as jpeg_im:
            img_pil.save(jpeg_im, format='JPEG')
            ex = make_example(jpeg_im.getvalue(), nx, ny, index, nframes)
        return index, ex.SerializeToString()
    
    def worker(qin,qout,d):
        try:
            while True:
                item = qin.get()
                if item is None:
                    break
                index, image = item
                assert index % num_worker == d, 'item worker %d not match %d'%(d,index)
                index, ex = process_image(index, image)
                qout.put((index, ex))
        except KeyboardInterrupt:
            pass
        finally:
            #print('finish output buffer %d'%d)
            qout.put(None)

    def write_worker(qout,split):
        writer = None
        try:
            i = 0
            while True:
                item = qout[i % num_worker].get()
                if item is None:
                    break
                index, ex = item
                assert index == i, 'index not match %d != %d'%(index, i)
                
                if index % split == 0:
                    if writer:
                        writer.close()
                    filename = os.path.join(dest, videoname, 
                            '%08d.tfrecord'%(index//split))
                    #print('file %s'%filename)
                    writer = tf.io.TFRecordWriter(filename) 
                writer.write(ex)
                i += 1
        except KeyboardInterrupt:
            pass
        finally:
            if writer:
                writer.close()

    jobs = []
    jobs2 = []
    q_in = [Queue(100) for i in range(num_worker)]
    q_out = [Queue(10000) for i in range(num_worker)]

    for i in range(num_worker):
        j1 = multiprocessing.Process(target=worker,args=(q_in[i],q_out[i],i))
        j1.start()
        jobs.append(j1)

    j2 = multiprocessing.Process(target=write_worker,args=(q_out,50000))
    j2.start()
    jobs2.append(j2)

    try:
        for index in tqdm(range(nframes)):
            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_in[index % num_worker].put((index, frame))
            else:
                break
    finally:    
        #print('finish input buffer')
        for i in range(num_worker):
            q_in[i].put(None)

        print('wait input jobs')
        for j in jobs:
            j.join()

        print('wait output jobs')
        for j in jobs2:
            j.join()
        
        print('done.')

####################################################
# Loading data, and defining model folder
####################################################

def analyze_videos(config,videos,videotype='avi',shuffle=1,trainingsetindex=0,gputouse=None,save_as_csv=False, destfolder=None,cropping=None):
    """
    Makes prediction based on a trained network. The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

    You can crop the video (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file. The same cropping parameters will then be used for creating the video.
    Note: you can also pass cropping = [x1,x2,y1,y2] coordinates directly, that then will be used for all videos. You can of course loop over videos & pass specific coordinates for each case.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    Examples
    --------

    Windows example for analyzing 1 video
    >>> deeplabcut.analyze_videos('C:\\myproject\\reaching-task\\config.yaml',['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'])
    --------

    If you want to analyze only 1 video
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi'])
    --------

    If you want to analyze all videos of type avi in a folder:
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos'],videotype='.avi')
    --------

    If you want to analyze multiple videos
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'])
    --------

    If you want to analyze multiple videos with shuffle = 2
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'], shuffle=2)

    --------
    If you want to analyze multiple videos with shuffle = 2 and save results as an additional csv file too
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml',['/analysis/project/videos/reachingvideo1.avi','/analysis/project/videos/reachingvideo2.avi'], shuffle=2,save_as_csv=True)
    --------

    """
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

    if gputouse is not None: #gpu selection
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    TF.logging.set_verbosity(TF.logging.WARN)

    if destfolder:
        tf.io.gfile.makedirs(destfolder)

    start_path=os.getcwd() #record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)

    if cropping is not None:
        cfg['cropping']=True
        cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']=cropping
        print("Overwriting cropping parameters:", cropping)
        print("These are used for all videos, but won't be save to the cfg file.")

    trainFraction = cfg['TrainingFraction'][trainingsetindex]

    project_path = cfg['project_path']
    trainbase_path = cfg['train_base_path']
    modelfolder=os.path.join(project_path,str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
    tmodelfolder=os.path.join(trainbase_path,str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

    if project_path != trainbase_path:
        print('copy to training folder...')
        resultfrom_path = os.path.join(str(modelfolder), 'train')
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

    trainpath = os.path.join(str(modelfolder), 'train')
    ttrainpath = os.path.join(str(tmodelfolder), 'train')
    
    # Check which snapshots are available and sort them by # iterations
    Snapshots = np.array([fn.split('.')[-2]for fn in os.listdir(trainpath) if "index" in fn])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    #dlc_cfg = read_config(os.path.join(modelfolder,'pose_cfg.yaml'))
    #dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
    SNP=Snapshots[-1]
    trainingsiterations = (SNP.split(os.sep)[-1]).split('-')[-1]

    print("Using %s" % trainingsiterations, "for model", tmodelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    #update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size']=cfg['batch_size']

    # update number of outputs
    dlc_cfg['num_outputs'] = cfg.get('num_outputs', 1)

    print('num_outputs = ', dlc_cfg['num_outputs'])
    
    # Name for scorer:
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)

    xyz_labs_orig = ['x', 'y', 'likelihood']
    suffix = [str(s+1) for s in range(dlc_cfg['num_outputs'])]
    suffix[0] = '' # first one has empty suffix for backwards compatibility
    xyz_labs = [x+s for s in suffix for x in xyz_labs_orig]

    pdindex = pd.MultiIndex.from_product([[DLCscorer],
                                          dlc_cfg['all_joints_names'],
                                          xyz_labs],
                                         names=['scorer', 'bodyparts', 'coords'])

    ##################################################
    # Datafolder
    ##################################################
    Videos=auxiliaryfunctions.Getlistofvideos(videos,videotype)

    if len(Videos)>0:
        #looping over videos
        for video in Videos:
            AnalyzeVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,ttrainpath,pdindex,save_as_csv, destfolder)

        os.chdir(str(start_path))
        print("The videos are analyzed. Now your research can truly start! \n You can create labeled videos with 'create_labeled_video'.")
        print("If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract any outlier frames!")
    else:
        print("No video was found in the path/ or single video with path:", videos)
        print("Perhaps the videotype is distinct from the videos in the path, I was looking for:",videotype)

    return DLCscorer

    
def GetPose(cfg,dlc_cfg,dlc,cap,nframes,batchsize):
    ''' Batchwise prediction of pose '''

    import threading
    import queue

    PredicteData = np.zeros((nframes, dlc_cfg['num_outputs'] * 3 * len(dlc_cfg['all_joints_names'])))
    ny,nx=int(cap.get(4)),int(cap.get(3))

    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    q = queue.Queue(4*dlc_cfg.batch_size)
    def worker():
        print('worker start')
        def make_input():
            while True:
                item = q.get()
                if item is None:
                    break
                im, count = item
                yield im.astype(np.float32), 0
                q.task_done()

        def predict_input_fn(batch_size=1):
            print('batch_size', batch_size)
            dataset = tf.data.Dataset.from_generator(make_input, 
                    (tf.float32,tf.float32), 
                    (tf.TensorShape([ny,nx,3]), tf.TensorShape([])))
            return dataset.batch(batch_size).prefetch(batch_size)

        tpu = os.getenv('TPU', False)
        if tpu:
            predictions = dlc.predict(
                    input_fn=lambda params: predict_input_fn(params["batch_size"]))
        else:
            predictions = dlc.predict(
                    input_fn=lambda: predict_input_fn(dlc_cfg.batch_size))
       
        image_count = 0
        try:
            for predicted in predictions:
                PredicteData[image_count, 0::3] = predicted['x'].flatten()
                PredicteData[image_count, 1::3] = predicted['y'].flatten()
                PredicteData[image_count, 2::3] = predicted['p'].flatten()
                image_count += 1
        except KeyboardInterrupt:
            pass
        print('worker_end')


    t = threading.Thread(target=worker)
    t.start()

    pbar=tqdm(total=nframes)
    counter = 0
    try:
        while(cap.isOpened()):
            pbar.update(1)
            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg['cropping']:
                    frame = img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
                else:
                    frame = img_as_ubyte(frame)
                
                q.put((frame, counter))
            else:
                nframes = counter
                print("Detected frames: ", nframes)
                break
            counter += 1
    except KeyboardInterrupt:
        q.join()
        # stop workers
        q.put(None)
        t.join()
        pbar.close()
        raise KeyboardInterrupt

    # block until all tasks are done
    q.join()
    # stop workers
    q.put(None)
    t.join()

    pbar.close()

    return PredicteData[:nframes],nframes

def copy_worker(temp_path, filepath):
    dst = os.path.join(temp_path,Path(filepath).name)
    tf.io.gfile.copy(filepath, dst)
    print(dst)
    return dst

def GetPoseTFR(cfg,dlc_cfg,dlc,record_files,nx,ny,nframes,batchsize):
    ''' Batchwise prediction of pose '''

    import threading
    import queue
    import uuid
    import re

    tpu = os.getenv('TPU', False)
    if tpu and not record_files[0].startswith('gs://'):
        trainbase_path = cfg['train_base_path']
        temp_path = os.path.join(trainbase_path, str(uuid.uuid4()))
        tf.io.gfile.makedirs(temp_path)
   
        print('copy tfrecord files...')
        with Pool(processes=8) as pool:
            record_files = pool.starmap(copy_worker, zip([temp_path]*len(record_files), record_files))

        print('copy done.')
    else:
        temp_path = None
    record_files.sort()
    PredicteData = np.zeros((nframes, dlc_cfg['num_outputs'] * 3 * len(dlc_cfg['all_joints_names'])))

    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    def predict_input_fn(batch_size=1):    
        def read_tfrecord(serialized):
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'frame_no': tf.io.FixedLenFeature([], tf.int64),
                    'nframes': tf.io.FixedLenFeature([], tf.int64),
                    'im_raw': tf.io.FixedLenFeature([], tf.string),
                })
            image = tf.io.decode_jpeg(features['im_raw'], channels=3)
            image.set_shape([ny, nx, 3])
            image = tf.image.convert_image_dtype(image, tf.float32) * 255.
            return image, tf.constant(0, tf.float32)

        dataset = (tf.data.Dataset.from_tensor_slices(record_files)
                .flat_map(tf.data.TFRecordDataset)
                .map(read_tfrecord, num_parallel_calls=batch_size*4)
                .prefetch(10000))
        dataset = dataset.batch(batch_size).prefetch(10000)
        return dataset

    if tpu:
        predictions = dlc.predict(
                input_fn=lambda params: predict_input_fn(params["batch_size"]))
    else:
        predictions = dlc.predict(
                input_fn=lambda: predict_input_fn(dlc_cfg.batch_size))
       
    pbar=tqdm(total=nframes)
    image_count = 0
    try:
        for predicted in predictions:
            pbar.update(1)
            if image_count < nframes:
                PredicteData[image_count, 0::3] = predicted['x'].flatten()
                PredicteData[image_count, 1::3] = predicted['y'].flatten()
                PredicteData[image_count, 2::3] = predicted['p'].flatten()
            else:
                print('more frames detected. %d/%d'%(image_count, nframes))
            image_count += 1
    except KeyboardInterrupt:
        pass

    pbar.close()
    if not temp_path is None:
        tf.io.gfile.rmtree(temp_path)

    return PredicteData[:nframes],nframes


def GetPoseF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,batchsize):
    ''' Batchwise prediction of pose '''

    PredicteData = np.zeros((nframes, dlc_cfg['num_outputs'] * 3 * len(dlc_cfg['all_joints_names'])))
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at
    ny,nx=int(cap.get(4)),int(cap.get(3))
    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte') # this keeps all frames in a batch
    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))
    while(cap.isOpened()):
            if counter%step==0:
                pbar.update(step)
            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg['cropping']:
                    frames[batch_ind] = img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
                else:
                    frames[batch_ind] = img_as_ubyte(frame)

                if batch_ind==batchsize-1:
                    pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
                    PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    batch_ind = 0
                    batch_num += 1
                else:
                   batch_ind+=1
            else:
                nframes = counter
                print("Detected frames: ", nframes)
                if batch_ind>0:
                    pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                    PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
                break
            counter+=1

    pbar.close()
    return PredicteData,nframes


def GetPoseS(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes):
    ''' Non batch wise pose estimation for video cap.'''
    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')
    
    PredicteData = np.zeros((nframes, dlc_cfg['num_outputs'] * 3 * len(dlc_cfg['all_joints_names'])))

    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))
    while(cap.isOpened()):
            if counter%step==0:
                pbar.update(step)

            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg['cropping']:
                    frame= img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
                else:
                    frame = img_as_ubyte(frame)
                pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
                PredicteData[counter, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
            else:
                nframes=counter
                break
            counter+=1

    pbar.close()
    return PredicteData,nframes


def AnalyzeVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,trainpath,pdindex,save_as_csv, destfolder=None):
    ''' Helper function for analyzing a video '''
    print("Starting to analyze % ", video)
    tf_record = video.endswith('.tfrecord')
    if tf_record:
        vname = Path(video).parents[0].stem
    else:
        vname = Path(video).stem
    if destfolder is None:
        if tf_record:
            destfolder = str(Path(video).parents[1])
        else:
            destfolder = str(Path(video).parents[0])
    dataname = os.path.join(destfolder,vname + DLCscorer + '.h5')
    try:
        # Attempt to load data...
        pd.read_hdf(dataname)
        print("Video already analyzed!", dataname)
    except FileNotFoundError:
        print("Loading ", video)

        if tf_record:
            def read_tfrecord_size(serialized):
                features = tf.io.parse_single_example(
                    serialized,
                    features={
                        'width': tf.io.FixedLenFeature([], tf.int64),
                        'height': tf.io.FixedLenFeature([], tf.int64),
                        'frame_no': tf.io.FixedLenFeature([], tf.int64),
                        'nframes': tf.io.FixedLenFeature([], tf.int64),
                        'im_raw': tf.io.FixedLenFeature([], tf.string),
                    })
                return features['width'], features['height'], features['nframes']

            record_files = tf.io.gfile.glob(video)
            dataset = tf.data.Dataset.from_tensor_slices(record_files).flat_map(tf.data.TFRecordDataset).map(read_tfrecord_size)
            with TF.Session() as sess:
                data = TF.data.make_one_shot_iterator(dataset).get_next()
                nx, ny, nframes = sess.run(data)

            print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)
            fps = 0
        else:
            cap=cv2.VideoCapture(video)

            fps = cap.get(5) #https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
            nframes = int(cap.get(7))
            duration=nframes*1./fps
            size=(int(cap.get(4)),int(cap.get(3)))

            ny,nx=size
            print("Duration of video [s]: ", round(duration,2), ", recorded with ", round(fps,2),"fps!")
            print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)
        
        start = time.time()

        model_fn = lambda features, labels, mode, params: pose_net(dlc_cfg).model_fn(features, labels, mode, params)

        tpu = os.getenv('TPU', False)
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
                        'width': nx,
                        'height': ny,
                        'stride': dlc_cfg.stride,
                        'num_outputs': dlc_cfg['num_outputs'],
                        },
                    config=run_config)
        else:
            dlc = TF.estimator.Estimator(
                    model_fn=model_fn,
                    model_dir=trainpath,
                    params={
                        'tpu': tpu,
                        'width': nx,
                        'height': ny,
                        'stride': dlc_cfg.stride,
                        'num_outputs': dlc_cfg['num_outputs'],
                        },
                    )

        if tf_record:
            print("Starting to extract posture")
            PredicteData,nframes=GetPoseTFR(cfg,dlc_cfg,dlc,record_files,nx,ny,nframes,int(dlc_cfg["batch_size"]))
        else:
            print("Starting to extract posture")
            PredicteData,nframes=GetPose(cfg,dlc_cfg,dlc,cap,nframes,int(dlc_cfg["batch_size"]))

        stop = time.time()

        if cfg['cropping']==True:
            coords=[cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']]
        else:
            coords=[0, nx, 0, ny]

        dictionary = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": DLCscorer,
            "DLC-model-config file": dlc_cfg,
            "fps": fps,
            "batch_size": dlc_cfg["batch_size"],
            "num_outputs": dlc_cfg["num_outputs"],
            "frame_dimensions": (ny, nx),
            "nframes": nframes,
            "iteration (active-learning)": cfg["iteration"],
            "training set fraction": trainFraction,
            "cropping": cfg['cropping'],
            "cropping_parameters": coords
        }
        metadata = {'data': dictionary}

        print("Saving results in %s..." %(destfolder))
        auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, range(nframes),save_as_csv)

def GetPosesofFrames(cfg,dlc_cfg, sess, inputs, outputs,directory,framelist,nframes,batchsize,rgb):
    ''' Batchwise prediction of pose  for framelist in directory'''
    from skimage import io
    print("Starting to extract posture")
    if rgb:
        im=io.imread(os.path.join(directory,framelist[0]),mode='RGB')
    else:
        im=io.imread(os.path.join(directory,framelist[0]))

    ny,nx,nc=np.shape(im)
    print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)

    PredicteData = np.zeros((nframes, dlc_cfg['num_outputs'] * 3 * len(dlc_cfg['all_joints_names'])))
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at

    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx,ny=cfg['x2']-cfg['x1'],cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(np.shape(im)[1]) and cfg['y1']>=0 and cfg['y2']<int(np.shape(im)[0]):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))

    if batchsize==1:
        for counter,framename in enumerate(framelist):
                #frame=io.imread(os.path.join(directory,framename),mode='RGB')
                if rgb:
                    im=io.imread(os.path.join(directory,framename),mode='RGB')
                else:
                    im=io.imread(os.path.join(directory,framename))

                if counter%step==0:
                    pbar.update(step)

                if cfg['cropping']:
                    frame= img_as_ubyte(im[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2'],:])
                else:
                    frame = img_as_ubyte(im)

                pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
                PredicteData[counter, :] = pose.flatten()
    else:
        frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte') # this keeps all the frames of a batch
        for counter,framename in enumerate(framelist):
                if rgb:
                    im=io.imread(os.path.join(directory,framename),mode='RGB')
                else:
                    im=io.imread(os.path.join(directory,framename))

                if counter%step==0:
                    pbar.update(step)

                if cfg['cropping']:
                    frames[batch_ind] = img_as_ubyte(im[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2'],:])
                else:
                    frames[batch_ind] = img_as_ubyte(im)

                if batch_ind==batchsize-1:
                    pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                    PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    batch_ind = 0
                    batch_num += 1
                else:
                   batch_ind+=1

        if batch_ind>0: #take care of the last frames (the batch that might have been processed)
            pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
            PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]

    pbar.close()
    return PredicteData,nframes,nx,ny


def analyze_time_lapse_frames(config,directory,frametype='.png',shuffle=1,trainingsetindex=0,gputouse=None,save_as_csv=False,rgb=True):
    """
    Analyzed all images (of type = frametype) in a folder and stores the output in one file.

    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    rbg: bool, optional.
        Whether to load image as rgb; Note e.g. some tiffs do not alow that option in io.imread, then just set this to false.

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1', frametype='.bmp')
    --------

    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png
    """
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

    TF.reset_default_graph()
    start_path=os.getcwd() #record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
    except FileNotFoundError:
      raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex=cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]

    #update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = cfg['batch_size'] 
    
    # Name for scorer:
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    # update number of outputs and adjust pandas indices
    dlc_cfg['num_outputs'] = cfg.get('num_outputs', 1)

    xyz_labs_orig = ['x', 'y', 'likelihood']
    suffix = [str(s+1) for s in range(dlc_cfg['num_outputs'])]
    suffix[0] = '' # first one has empty suffix for backwards compatibility
    xyz_labs = [x+s for s in suffix for x in xyz_labs_orig]

    pdindex = pd.MultiIndex.from_product([[DLCscorer],
                                          dlc_cfg['all_joints_names'],
                                          xyz_labs],
                                         names=['scorer', 'bodyparts', 'coords'])

    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    ##################################################
    # Loading the images
    ##################################################
    #checks if input is a directory
    if os.path.isdir(directory)==True:
        """
        Analyzes all the frames in the directory.
        """
        print("Analyzing all frames in the directory: ", directory)
        os.chdir(directory)
        framelist=np.sort([fn for fn in os.listdir(os.curdir) if (frametype in fn)])

        vname = Path(directory).stem
        dataname = os.path.join(directory,vname + DLCscorer + '.h5')
        try:
            # Attempt to load data...
            pd.read_hdf(dataname)
            print("Frames already analyzed!", dataname)
        except FileNotFoundError:
            nframes = len(framelist)
            if nframes>1:
                start = time.time()

                PredicteData,nframes,nx,ny=GetPosesofFrames(cfg,dlc_cfg, sess, inputs, outputs,directory,framelist,nframes,dlc_cfg['batch_size'],rgb)
                stop = time.time()

                if cfg['cropping']==True:
                    coords=[cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']]
                else:
                    coords=[0, nx, 0, ny]

                dictionary = {
                    "start": start,
                    "stop": stop,
                    "run_duration": stop - start,
                    "Scorer": DLCscorer,
                    "config file": dlc_cfg,
                    "batch_size": dlc_cfg["batch_size"],
                    "num_outputs": dlc_cfg["num_outputs"],
                    "frame_dimensions": (ny, nx),
                    "nframes": nframes,
                    "cropping": cfg['cropping'],
                    "cropping_parameters": coords
                }
                metadata = {'data': dictionary}

                print("Saving results in %s..." %(directory))

                auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, framelist,save_as_csv)
                print("The folder was analyzed. Now your research can truly start!")
                print("If the tracking is not satisfactory for some frome, consider expanding the training set.")
            else:
                print("No frames were found. Consider changing the path or the frametype.")

    os.chdir(str(start_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('config')
    cli_args = parser.parse_args()
