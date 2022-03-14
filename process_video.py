import cv2
import os
import glob
from cv2 import rotate
import mediapipe as mp
import numpy as np
import pickle as pkl
from tqdm import tqdm
import multiprocessing

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process(video_file):
    # loop through each video file

    # load the video file into memory using OpenCV
    video = cv2.VideoCapture(video_file)

    # check if the video file has been processed already
    pkl_f = "pkls/" + video_file.split('\\')[-1].split('.')[0] + "pointer.pkl"
    if os.path.exists(pkl_f):
        print(f"Skipping {video_file}")
        return

    # finger_tip_pos
    finger_tip_pos = []

    # get the number of frames in the video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in tqdm(range(num_frames)):
        # get the next frame
        ret, frame = video.read()
            
        # check if there is a frame
        if not ret: continue
            
        # rotate the frame if needed
        if np.shape(frame)[0] > np.shape(frame)[1]:
            frame = rotate_frame(frame)

        # use mediapipe to get the hand positions
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7) as hands:
            # convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # print handedness (left v.s. right hand).
            #print(f'Handedness of {video_file}:')
            #print(results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue

            # draw hand landmarks of each hand.
            #print(f'Hand landmarks of {video_file}:')
            image_hight, image_width, _ = frame.shape
            annotated_image = cv2.flip(frame.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                # print index finger tip coordinates.
                '''
                print(
                    f'Index finger tip coordinate: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                    
                )
                '''
                frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                #print(f"{frame_number}")
                finger_tip_pos.append((frame_number,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight))
                
                """
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                """
                
            # resize_and_show(cv2.flip(annotated_image, 1))
            #cv2.putText(annotated_image, f'frame {int(video.get(cv2.CAP_PROP_POS_FRAMES))}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.imshow("hands", annotated_image)
            #cv2.waitKey(1)
            
        
        # check if the user wants to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # pickle finger_tip_pos and save it
    fname = "pkls/" + video_file.split("\\")[-1][:-4] + ".pkl"
    pkl.dump(finger_tip_pos, open(fname, "wb"))
    print(f'Saved file {fname}')


def rotate_frame(frame):
    # rotate the incoming frame by 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def main():
    # get the list of video files in the videos folder
    video_files = glob.glob("videos/*.mp4")
    with multiprocessing.Pool() as pool:
        pool.map(process, video_files)

if __name__ == "__main__":
    main()