from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle as pkl

def main():
    pkl_files = glob.glob("pkls/*.pkl")
    for pkl_file in pkl_files:
        data_raw = pkl.load(open(pkl_file, "rb"))
        print(f'Loaded file {pkl_file}')

        data = np.array(data_raw)
        ax = plt.axes(projection ='3d')

        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Time (frame)')
        ax.set_title(f'Trajectory of {pkl_file}')

        ax.scatter(data[:,1], data[:,2], data[:,0], label=f'Trajectory: {pkl_file}')

        # scatter plot any data outside of the limits as a red dot
        #ax.scatter(data[:,1][data[:,0] > data[-1,0]], data[:,2][data[:,0] > data[-1,0]], data[:,0][data[:,0] > data[-1,0]], c='r', label='Out of bounds')
        plt.show()

if __name__ == "__main__":
    main()