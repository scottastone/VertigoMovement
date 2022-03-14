import cv2
import pickle as pkl
import numpy as np
import glob
from tqdm import tqdm
import os
import multiprocessing

def process(pkl_file, video_file):
    data = pkl.load(open(pkl_file, "rb"))
    print(f'Loaded file {pkl_file}')
    frame_idx, pos = get_frame_and_positions(data)

    # check if the video file has been processed already
    pkl_f = "demo_videos/" + video_file.split('\\')[-1].split('.')[0] + "pointer.mp4"
    if os.path.exists(pkl_f):
        print(f"Skipping {video_file}")
        return

    video = cv2.VideoCapture(video_file)
    print(f'Loaded video {video_file}')

    # write video file to the demo_videos folder
    # make a video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_name = pkl_file.split("\\")[-1].split(".")[0]
    video_out_name = f'demo_videos/{video_name}.mp4'
    out = cv2.VideoWriter(video_out_name, fourcc, 480.0, (1280, 720))
    out.open(video_out_name, fourcc, 120.0, (1280, 720))
    print(f"Opened writer object {video_out_name}")

    # get the number of frames in the video_file
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(num_frames)):
        ret, frame = video.read()
        if not ret: continue
            
        # if the dimensions are not 16:9, rotate the frame
        if np.shape(frame)[0] > np.shape(frame)[1]:
            frame = rotate_frame(frame)
            
        active_frames_idx = frame_num >= frame_idx
        pos_to_draw = pos[active_frames_idx]

        for p in pos_to_draw:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        
        # add the current frame number with second conversion to the top left of the screen
        sec_passed = frame_num / 480.0
        cv2.putText(frame, f"{frame_num}: {sec_passed:0.2f}s" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.imshow("Frame", frame)
        # show the frame
        #cv2.waitKey(1)

        out.write(frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Saved video {video_file}')


def get_frame_and_positions(data):
    frames, positions = [], []
    for frame, pos_x, pos_y in data:
        frames.append(frame-1)
        positions.append([pos_x, pos_y])

    return np.array(frames), np.array(positions)

def rotate_frame(frame):
    # rotate the incoming frame by 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return frame

if __name__ == "__main__":
    pkl_files = glob.glob("pkls/*.pkl")
    video_files = glob.glob("videos/*.mp4")

    with multiprocessing.Pool() as pool:
        pool.starmap(process, zip(pkl_files, video_files))

    