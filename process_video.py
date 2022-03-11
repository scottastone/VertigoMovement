import cv2
import os
import glob
from cv2 import rotate
import mediapipe as mp
import numpy as np
import pickle as pkl

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def rotate_frame(frame):
    # rotate the incoming frame by 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def main():
    # get the list of video files in the videos folder
    video_files = glob.glob("videos/*.mp4")


    # loop through each video file
    for video_file in video_files:
        # load the video file into memory using OpenCV
        video = cv2.VideoCapture(video_file)

        # finger_tip_pos
        finger_tip_pos = []

        while True:
            # get the next frame
            ret, frame = video.read()
                
            # check if there is a frame
            if not ret:
                # pickle finger_tip_pos and save it
                fname = video_file.split("\\")[-1][:-4] + "pointer.pkl"
                pkl.dump(finger_tip_pos, open(fname, "wb"))
                print(f'Saved file {fname}')
                break

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
                    print(f"{frame_number}")
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

    pass

if __name__ == "__main__":
    main()