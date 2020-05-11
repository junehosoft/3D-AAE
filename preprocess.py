import numpy as np
import cv2
import os
import glob
import pywavefront
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import *
from binvox_rw import read_as_3d_array

# convert vertices list to triplets, and create voxels
def world_to_voxels(verts):
    verts = np.reshape(np.array(verts), (-1, 3))
    maxDiff = 0
    voxel = np.zeros((64, 64, 64))
    for i in range(3):
        diff = max(verts[:,i]) - min(verts[:,i])
        # print(max(verts[:,i]), min(verts[:,i]), diff)
        maxDiff = max((maxDiff, diff))
        verts[:, i] = verts[:, i] - min(verts[:, i]) - diff / 2
    for i in range(3):
        verts[:, i] = verts[:, i] * (63 / maxDiff) + 32
    for v in verts:
        voxel[int(v[0]), int(v[2]), int(v[1])] = 1 
    return voxel
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.voxels(voxel, facecolors='red')
    #plt.show()
    
        

if __name__=="__main__":
    #x, y, z = np.indices((8, 8, 8))
    # draw cuboids in the top left and bottom right corners, and a link between them
    #cube1 = (x < 3) & (y < 3) & (z < 3)
    #print(cube1)
    
    #path_names = []
    for i in range(len(CATEGORIES)):
        cat = CATEGORIES[i]
        path_names = (glob.glob('data/IKEA/IKEA_{}_*/*.obj'.format(cat)))
        count = 0
        for path in path_names:
            cmd = 'binvox -d 64 -cb {}'.format(path)
            os.system(cmd)
            output = '.'.join([path[:-4], 'binvox'])
            with open(output, 'rb') as f:
                m1 = read_as_3d_array(f)
                np.save('data/new_voxels/{}_{}.npy'.format(cat, count), m1.data)
                if count < 1:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    ax.voxels(m1.data, facecolors='red')
                    plt.axis('off')
                    plt.savefig("data/new_voxels/{}_{}.png".format(cat, count), bbox_inches='tight')
                    plt.close()
                count += 1
            '''with open(path) as dataFile:
                data = dataFile.read()
                obj = data[data.find('{') : data.rfind('}')+1]
                jsonObj = json.loads(obj)
                voxel = world_to_voxels(jsonObj['vertices'])
                np.save('data/voxels/{}_{}.npy'.format(cat, count), voxel)
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.voxels(voxel, facecolors='red')
                plt.axis('off')
                plt.savefig("data/voxels/{}_{}.png".format(cat, count), bbox_inches='tight')
                plt.close()
                count += 1'''
