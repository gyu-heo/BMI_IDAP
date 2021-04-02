## import camera GPIO pin signals

### IMPORT VIDEOS
## Define DIRECTORY of video(s) to use and IMPORT videos (as read objects) into openCV
## Be careful to follow input the directories properly below. Input directory
## and file name (or file name prefix) in either the m

import os
import cv2
import numpy as np

def getVideoDirectoryProperties(multiple_files_pref , dir_vid , fileName_vid_prefix , fileName_vid_suffix , fileName_vid , slash_type , print_fileNames_pref):
    """
    gets video directory properties
    Args:
        multiple_files_pref (bool): whether or not you are looking for multiple files
        dir_vid (string): path to video(s)
    Returns:
    """

    # # This option imports all of the videos with a defined file name prefix in a folder
    # # OR just imports a single defined file
    # multiple_files_pref = 1

    # dir_vid = r'/media/rich/bigSSD RH/res2p/Camera data/round 4 experiments/mouse 6.28/20201102/cam1'
    # # dir_vid = r'/media/rich/bigSSD RH/res2p/Camera data/round 4 experiments/mouse 6.28/20201102/cam1'

    # # Used only if 'multiple_files_pref'==1
    # fileName_vid_prefix = 'cam1_2020-11-02-185734-' 
    # fileName_vid_suffix = '.avi'

    # # Used only if 'multiple_files_pref'==0
    # fileName_vid = 'gmou06_082720_faceTrack_session1_DeInter100.avi'


    ### == IMPORT videos ==
    print_fileNames_pref = 1

    if multiple_files_pref:
        ## first find all the files in the directory with the file name prefix
        fileNames_allInPathWithPrefix = []
        for ii in os.listdir(dir_vid):
            if os.path.isfile(os.path.join(dir_vid,ii)) and fileName_vid_prefix in ii:
                fileNames_allInPathWithPrefix.append(ii)
        numVids = len(fileNames_allInPathWithPrefix)
        
        ## make a variable containing all of the file paths
        path_vid_allFiles = list()
        for ii in range(numVids):
            path_vid_allFiles.append(f'{dir_vid}{slash_type}{fileNames_allInPathWithPrefix[ii]}')
            
    else: ## Single file import
        path_vid = f'{dir_vid}{slash_type}{fileName_vid}'
        path_vid_allFiles = list()
        path_vid_allFiles.append(path_vid)
        numVids = 1
    path_vid_allFiles = sorted(path_vid_allFiles)
            
    ## get info on the imported video(s): num of frames, video height and width, framerate
    if multiple_files_pref:
        path_vid = path_vid_allFiles[0]
        video = cv2.VideoCapture(path_vid_allFiles[0])
        numFrames_firstVid = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        numFrames_allFiles = np.ones(numVids) * np.nan # preallocation
        for ii in range(numVids):
            video = cv2.VideoCapture(path_vid_allFiles[ii])
            numFrames_allFiles[ii] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        numFrames_total_rough = np.uint64(sum(numFrames_allFiles))
            
        print(f'number of videos: {numVids}')
        print(f'number of frames in FIRST video (roughly):  {numFrames_firstVid}')
        print(f'number of frames in ALL videos (roughly):   {numFrames_total_rough}')
    else:
        video = cv2.VideoCapture(path_vid_allFiles[0])
        numFrames_onlyVid = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        numFrames_total_rough = numFrames_onlyVid
        numFrames_allFiles = numFrames_total_rough
        print(f'number of frames in ONLY video:   {numFrames_onlyVid}')
        

    Fs = video.get(cv2.CAP_PROP_FPS) ## Sampling rate (FPS). Manually change here if necessary
    print(f'Sampling rate pulled from video file metadata:   {round(Fs,3)} frames per second')
        
    if print_fileNames_pref:
        print(f'\n {np.array(path_vid_allFiles).transpose()}')

        
    video.set(1,1)
    ok, frame = video.read()
    vid_height = frame.shape[0]
    vid_width = frame.shape[1]

    return path_vid_allFiles , vid_height , vid_width , numFrames_total_rough , Fs