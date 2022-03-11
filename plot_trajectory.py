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
        fig = plt.figure()
        ax = plt.axes(projection ='3d')

        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Time')
        ax.scatter(data[:,1], data[:,2], data[:,0], label=f'Trajectory: {pkl_file}')

        #plt.plot(data[:,0], data[:,1], 'b')
        #plt.plot(data[:, 0], data[:, 2], 'r')
        plt.show()



        pass


if __name__ == "__main__":
    main()