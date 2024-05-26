import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json
import logging
from glob import glob
from os.path import join
from pickle import FALSE
import os
import pandas as pd
import hydra
import numpy as np
from google.protobuf.json_format import MessageToDict
from matplotlib.animation import FuncAnimation
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_python_vision
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
log = logging.getLogger(__name__)
from scipy import interpolate

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM,SimpleRNN,GRU,Dropout,Flatten,Input
from keras.preprocessing.sequence import pad_sequences

from myProject.settings import BASE_DIR

import tensorflow as tf

model_path= BASE_DIR + '/myApp/util/model99.h5'
# inference_layer = tf.keras.layers.TFSMLayer(model_path , call_endpoint='serving_default')

# # Create a Sequential model and add the inference layer
# model = tf.keras.Sequential([inference_layer])
model=tf.keras.models.load_model(model_path)


NOSE = 0
NECK = 1
RSHO = 2
RELB = 3
RWRI = 4
LSHO = 5
LELB = 6
LWRI = 7
MHIP = 8
RHIP = 9
RKNE = 10
RANK = 11
LHIP = 12
LKNE = 13
LANK = 14
REYE = 15
LEYE = 16
REAR = 17
LEAR = 18
LBTO = 19
LSTO = 20
LHEL = 21
RBTO = 22
RSTO = 23
RHEL = 24

total_steps=0
distance_covered=0
avg_step_length=0
valid_frames = 0



def fill_nan(A):
    inds = np.arange(A.shape[0]) 
    good = np.where(np.isfinite(A))
    if(len(good[0]) <= 1):
        return A
   
    # linearly interpolate and then fill the extremes with the mean (relatively similar to)
    # what kalman does 
    f = interpolate.interp1d(inds[good], A[good],kind="linear",bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    B = np.where(np.isfinite(B),B,np.nanmean(B))
    return B
    
def impute_frames(frames):
    return np.apply_along_axis(fill_nan,arr=frames,axis=0)
def convert_json2csv(json_directory,personID):
    # determine the number of frames
    nframes = len(os.listdir(json_directory))
    print(nframes)
  
    # initialize res to be array of NaN
    res = np.zeros((nframes,75))
    res[:] = np.nan
    
    # read in JSON files
    for frame in range(0,nframes):
        test_image_json = '%s_%s_keypoints.json' %            (str(personID), str(frame))
        test_image_json = os.path.join('kp/',test_image_json)
        if not os.path.isfile(test_image_json):
            print('NO')
            break
        with open(test_image_json) as data_file:  
            data = json.load(data_file)
        for person in data['people']:
            keypoints = person['pose_keypoints_2d']
            xcoords = [keypoints[i] for i in range(len(keypoints)) if i % 3 == 0]
            counter = 0
            res[frame,:] = keypoints
            break

    return res

def drop_confidence_cols(keypoint_array):
    num_parts = keypoint_array.shape[1]/3 # should give 25 (# of OpenPose keypoints)
    processed_cols = [True,True,False] * int(num_parts) 
    return keypoint_array[:,processed_cols]

def filter_frames(frames, sd=1):
    return np.apply_along_axis(lambda x: gaussian_filter1d(x,sd),
                               arr = frames,
                               axis = 0)
def find_mid_sd(signal_array):
    num_parts = signal_array.shape[1]/2 

    # Derive mid hip position
    mhip_x = ((signal_array[:,2*RHIP] + signal_array[:,2*LHIP])/2).reshape(-1,1)
    mhip_y = ((signal_array[:,2*RHIP+1] + signal_array[:,2*LHIP+1])/2).reshape(-1,1)
    mhip_coords = np.hstack([mhip_x,mhip_y]*int(num_parts))

    # Normalize to hip-knee distance
    topoint = lambda x: range(2*x,2*x+2)
    scale_vector_R = np.apply_along_axis(lambda x: np.linalg.norm(x[topoint(RHIP)] -
                                                                  x[topoint(RKNE)]),1,signal_array)
    scale_vector_L = np.apply_along_axis(lambda x: np.linalg.norm(x[topoint(LHIP)] -
                                                                  x[topoint(LKNE)]),1,signal_array)
    scale_vector = ((scale_vector_R + scale_vector_L)/2.0).reshape(-1,1)
    return mhip_coords, scale_vector

def get_angle(A,B,C,centered_filtered):
   
    # finds the angle ABC, assumes that confidence columns have been removed
    # A,B and C are integers corresponding to different keypoints
    
    p_A = np.array([centered_filtered[:,2*A],centered_filtered[:,2*A+1]]).T
    p_B = np.array([centered_filtered[:,2*B],centered_filtered[:,2*B+1]]).T
    p_C = np.array([centered_filtered[:,2*C],centered_filtered[:,2*C+1]]).T
    p_BA = p_A - p_B
    p_BC = p_C - p_B
    dot_products = np.sum(p_BA*p_BC,axis=1)
    norm_products = np.linalg.norm(p_BA,axis=1)*np.linalg.norm(p_BC,axis=1)
    return np.arccos(dot_products/norm_products)

def get_distance(A,B,centered_filtered):
    p_A = np.array([centered_filtered[:,2*A],centered_filtered[:,2*A+1]]).T
    p_B = np.array([centered_filtered[:,2*B],centered_filtered[:,2*B+1]]).T    
    p_BA = p_A - p_B
    return np.linalg.norm(p_BA,axis=1)

def peakdet(v, delta, x = None):
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        print('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        print('Input argument delta must be a scalar')
    
    if delta <= 0:
        print('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
# Calculate a similarity score between two curves (e.g., cosine similarity)
def similarity_score(curve1, curve2):
    return np.dot(curve1, curve2) / (np.linalg.norm(curve1) * np.linalg.norm(curve2))
    
def preProccessVideo(path):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Path to the input video
    video_path = path
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))+1
    global resul
    resul = np.zeros((nframes,75))
    resul[:] = np.nan
    frame_count=0
    personID=100

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fps = cap.get(cv2.CAP_PROP_FPS)
        # frame = cv2.resize(frame,(1024,800))
        # Convert frame to RGB (MediaPipe requires RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose detection
        results = pose.process(frame_rgb)


        height, width, _ = frame.shape
        # Draw pose landmarks on the frame
        new_lmList = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # 3D view of landmarks
            # qq = mp_drawing.plot_landmarks( landmarks_list,  mp_pose.POSE_CONNECTIONS)
        
            # Plot and save the landmarks on the image
            # mp_drawing.draw_landmarks(
            # 	frame,
            # 	results.pose_landmarks,
            # 	mp_pose.POSE_CONNECTIONS,
            # 	landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
            # # write image to storage
            # filename=frame.replace(".","_keypoints.")
            # cv2.imwrite(filename, frame)
        
            axes_weights=[1.0, 1.0, 0.2, 1.0]
            # scale the z dimension for all landmarks by 0.2
            temp = [landmarks[i].z * axes_weights[2] for i in range(len(landmarks))]
            # replace with the updated z value
            for i in range(len(landmarks)):
                landmarks[i].z = temp[i]
        
            tmp =[]
            onlyList = []
            list4json = []
        
            # converting the landmarks to a list
            for idx, coords in enumerate(landmarks):
                coords_dict = MessageToDict(coords)
                # print(coords_dict)
                qq = (coords_dict['x'], coords_dict['y'], coords_dict['visibility'])
                tmp.append(qq)
        
            #  SCALING the x and y coordinates with the resolution of the image to get px corrdinates
            for i in range(len(tmp)):
                tmp[i] = ( int(np.multiply(tmp[i][0], width)), \
                        int(np.multiply(tmp[i][1], height)), \
                        tmp[i][2])
            # Calculate the two additional joints for openpose and add them
            # NECK KPT
            tmp[1] = ( (tmp[11][0] - tmp[12][0]) / 2 + tmp[12][0], \
                    (tmp[11][1] + tmp[12][1]) / 2 , \
                    0.95 )
            # saving the hip mid point in the list for later use
            stash = tmp[8]
            # HIP_MID
            tmp.append(stash)
            tmp[8] = ( (tmp[23][0] - tmp[24][0]) / 2 + tmp[24][0], \
                    (tmp[23][1] + tmp[24][1]) / 2 , \
                    0.95 )
        
            # Reordering list to comply to openpose format
            # For the order table,refer to the Notion page
            # restoring the saved hip mid point
            mp_to_op_reorder = [0, 1, 12, 14, 16, 11, 13, 15, 8, 24, 26, 28, 23, 25, 27, 5, 2, 33, 7, 31, 31, 29, 32, 32, 30, 0, 0, 0, 0, 0, 0, 0, 0]
        
            onlyList = [tmp[i] for i in mp_to_op_reorder]
        
            # delete the last 8 elements to conform to OpenPose joint length of 25
            del onlyList[-8:]
            
        
            # OpenPose format requires only a list of all landmarkpoints. So converting to a simple list
            for nums in onlyList:
                for val in nums:
                    list4json.append(val)
                    # list4df.append(val)   
            # Making the JSON openpose format and adding the data
            json_data = {
                "version": 1.3,
                "people": [
                    {
                        "person_id"              : [-1],
                        "pose_keypoints_2d"      : list4json,
                        "face_keypoints_2d"      : [],
                        "hand_left_keypoints_2d" : [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d"      : [],
                        "face_keypoints_3d"      : [],
                        "hand_left_keypoints_3d" : [],
                        "hand_right_keypoints_3d": []
                    }
                ]
            }
            for person in json_data['people']:
                keypoints = person['pose_keypoints_2d']
                xcoords = [keypoints[i] for i in range(len(keypoints)) if i % 3 == 0]
                counter = 0
                resul[frame_count,:] = keypoints
        # 	json_filename = str(personID)+'_'+ str(frame_count) +"_keypoints" + ".json"
        # 	# json_filename = json_filename.replace(".png","_keypoints")
    
        # 	with open(os.path.join('kp', json_filename), 'w') as fl:
        # 		fl.write(json.dumps(json_data, indent=2, separators=(',', ': ')))
        # 	print(f"Writing JSON file: {json_filename}")
        # print(f'{len(onlyList)} \n frame no:{frame_count} \n')   
        if results.pose_landmarks:
            frame_count += 1
            
            # log.info(f"Writing JSON file: {json_filename}")


        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    valid_frames = frame_count
    # print(valid_frames)
    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    df=pd.DataFrame(resul) 
    df.dropna(how='all', inplace=True)
    res=df.to_numpy()

    PLOT_COLS = {
        "Right ankle": RANK,
        "Right knee": RKNE,
        "Right hip": RHIP,
    }   
    res_processed = res.copy()
    res_processed = drop_confidence_cols(res_processed)
    # if keypoint is not detected, OpenPose returns 0. For undetected keypoints,
    # we change their values to NaN. Wo do it for all dimensions of all keypoints
    # in all frames at once
    res_processed[res_processed < 0.0001] = np.NaN 

    # Note that we use res_processed < 0.0001 instead of res_processed == 0.0.
    # While in the case of this video it makes no difference, it is usually
    # a good practice in computer science to account for the fact that 
    # 0.0 might be actually actually represented as a number slightly greater than
    # 0. See more on this topic here
    # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

    res_processed = impute_frames(res_processed)
    res_processed = filter_frames(res_processed, 1)
    res_mhip_coords, res_scale_vector = find_mid_sd(res_processed)
    res_processed = (res_processed-res_mhip_coords)/res_scale_vector
    knee_angle = (np.pi - get_angle(RANK, RKNE, RHIP, res_processed))*180/np.pi
    dst_org = get_distance(RBTO, LBTO, res_processed) * np.sign(res_processed[:,LBTO*2] - res_processed[:,RBTO*2])
    dst = gaussian_filter1d(dst_org.copy(),5)
    maxs, mins = peakdet(dst, 0.5)

    #****Total distance  & total steps**
    total_steps=len(maxs)*2
    distance_covered=float(sum(maxs[1]))
    avg_step_length= float(sum(maxs[1]))/total_steps
    # print(f'total_steps:{total_steps} \nDistance Covered : {distance_covered} \nAvg step length: {avg_step_length} ')
    # print(len(maxs),len(mins))

    print("DONE: maximums and minimums identified")
    segments = []
    sstart = int(maxs[0][0])
    for i in range(1,len(maxs)):
        end = int(maxs[i][0])
        segments.append(knee_angle[sstart:end])
        print("Segment from {} to {}".format(sstart,end))
        sstart = end
    grid = np.linspace(0.0, 1.0, num=100)
    curves = np.zeros([len(segments),100])
    for i in range(len(segments)):
        org_space = np.linspace(0.0, 1.0, num=int(segments[i].shape[0]))
        f = interpolate.interp1d(org_space, segments[i], kind="linear")
        # plt.plot(grid*100, f(grid), linestyle="-", linewidth=2.5)
        # plt.legend(["Cycle {}".format(k) for k in range(len(segments))],loc=1)
        curves[i,:] = f(grid)
    reference_curve = np.mean(curves, axis=0)
    input_data = reference_curve.tolist()
    input_data = np.array(input_data)
    input_data = input_data.reshape(-1,1,100)
  # Reshape to have batch size 1 and 100 features
    predictions = model.predict(input_data)
    print(predictions)
    binary_predictions = (predictions > 0.7).astype(int)
    if binary_predictions[0]==1:
        return 1
    else:
        return 0
