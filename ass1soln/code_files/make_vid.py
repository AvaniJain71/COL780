import os
import cv2 as cv

data_path = '/Users/silentstorm/SEM6/COL780/COL780 Dataset/Candela_m1.10/final'
files = []
for i in range(350):
    file_name = os.path.join(data_path, '%06d' % i + '.png')
    files.append(file_name)
os.chdir(os.path.dirname(data_path))
img1 = cv.imread(files[10])
width = img1.shape[0]
height = img1.shape[1]

def generate_video():
    video = cv.VideoWriter('final.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (height, width))     
    for file in files:
        img = cv.imread(file)
        video.write(img) 
    video.release()

generate_video()