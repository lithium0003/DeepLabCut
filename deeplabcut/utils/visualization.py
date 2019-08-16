"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import numpy as np
import matplotlib as mpl
import platform
from pathlib import Path

if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
elif platform.system() == 'Darwin':
    mpl.use('WXAgg')
else:
    mpl.use('TkAgg') #TkAgg
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def MakeLabeledImage(im,labeled,predict,pcutoff,bodyparts,colors,cfg,labels=['+','.','x'],scaling=1):
    '''Creating a labeled image with the original human labels, as well as the DeepLabCut's! '''
    alphavalue=cfg['alphavalue'] #.5
    dotsize=cfg['dotsize'] #=15

    plt.axis('off')
    if np.ndim(im)>2: #color image!
        h,w,numcolors=np.shape(im)
    else:
        h,w=np.shape(im)
    plt.figure(frameon=False,figsize=(w*1./100*scaling,h*1./100*scaling))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.imshow(im,'gray')
    
    for bpindex,bp in enumerate(list(bodyparts)):
        if np.isfinite(predict[bp,0]+predict[bp,1]):
            x = int(predict[bp,0])
            y = int(predict[bp,1])
            p = predict[bp,2]
            if p > pcutoff:
                plt.plot(x,y,labels[1],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
            else:
                plt.plot(x,y,labels[2],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
        if np.isfinite(labeled[bp,0]+labeled[bp,1]):
            x = int(labeled[bp,0])
            y = int(labeled[bp,1])
            plt.plot(x,y,labels[0],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()


def PlottingandSaveLabeledFrame(imagename,im,labeled,predict,isTrain,cfg,colors,comparisonbodyparts,foldername):
    im /= 255.
    path_name, base_name = os.path.split(imagename)
    imagename,ext = os.path.splitext(base_name)
    imfoldername = os.path.basename(path_name)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    MakeLabeledImage(im,labeled,predict,cfg["pcutoff"],comparisonbodyparts,colors,cfg)

    if isTrain:
        full_path = os.path.join(foldername,'Training-'+imfoldername+'-'+imagename)
    else:
        full_path = os.path.join(foldername,'Test-'+imfoldername+'-'+imagename)
    
    # windows throws error if file path is > 260 characters, can fix with prefix. see https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
    if (len(full_path) >= 260) and (os.name == 'nt'):
        full_path = '\\\\?\\'+full_path
    plt.savefig(full_path)

    plt.close("all")
