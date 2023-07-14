import numpy as np
import cv2 as cv
import os
import skimage
from scipy import ndimage
from skimage import io, util
from skimage.color import rgb2gray

#print(os.path.dirname(os.path.abspath('Assignment2.ipynb')))
data_path = os.path.join(os.path.dirname(os.path.abspath('Assignment2.ipynb')), 'dataset/5')
output_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), 'result5')
final_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), 'final5')

number_files = len(os.listdir(data_path)) - 1
files = []
output = []
for i in range(number_files):
    file_name = os.path.join(data_path, 'image %d' % i + '.jpg')
    files.append(file_name)

#img1 = cv.imread(files[0])
#shape = img1.shape
#gray1 = rgb2gray(img1)
#print(shape)

#cv.imshow("test", gray1)
#cv.waitKey(0)
#cv.destroyAllWindows()
#x = 2
#filename = '%03d' % x + '.jpg'
#print(filename)
#cv.imwrite(os.path.join(output_path, filename), img1)

#def rgb2gray(img) :
    #return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


size = 5
thresh = 0.138
power = []

output_no = 0
for file in files:
        
    img = cv.imread(file)
    shape = img.shape
    
    filtered = ndimage.gaussian_filter(img, sigma = 1.0)
    out = img.copy()
    gray = rgb2gray(filtered)
    dy, dx = np.gradient(gray)
    
    ixx = dx ** 2
    iyy = dy ** 2
    ixy = dx * dy

    corners = []
    for i in range(1, shape[0]):
        ixx[i][0] += ixx[i - 1][0]
        iyy[i][0] += iyy[i - 1][0]
        ixy[i][0] += ixy[i - 1][0]
                
    for j in range(1, shape[1]):
        ixx[0][j] += ixx[0][j - 1]
        iyy[0][j] += iyy[0][j - 1]
        ixy[0][j] += ixy[0][j - 1]
                             
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            ixx[i][j] += ixx[i - 1][j] + ixx[i][j - 1] - ixx[i - 1][j - 1]
            iyy[i][j] += iyy[i - 1][j] + iyy[i][j - 1] - iyy[i - 1][j - 1]
            ixy[i][j] += ixy[i - 1][j] + ixy[i][j - 1] - ixy[i - 1][j - 1]
            
    for i in range(0, shape[0] - size):
        for j in range(0, shape[1] - size):
            Vxx = ixx[i + size][j + size] - ixx[i + size][j] - ixx[i][j + size] + ixx[i][j]
            Vyy = iyy[i + size][j + size] - iyy[i + size][j] - iyy[i][j + size] + iyy[i][j]
            Vxy = ixy[i + size][j + size] - ixy[i + size][j] - ixy[i][j + size] + ixy[i][j]
                
            det = Vxx * Vyy - Vxy * Vxy
            trace = Vxx + Vyy
            measure = det / trace
                
            if measure > thresh:
                corners.append([i, j, measure])
                #out[i][j] = [255, 0, 0]
    sortedc = sorted(corners, key = lambda x: x[2], reverse = True)
    final = []
    final.append(sortedc[0][:-1])
    dis = 2 * size
    for i in sortedc :
        for j in final :
            if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                break
        else :
            final.append(i[:-1])
    power.append(final)        
    for el in final:
        out[el[0]][el[1]] = [255, 0, 0]
        cv.rectangle(out, (int(el[1] - 8), int(el[0] - 8)), (int(el[1] + 8), int(el[0] + 8)), (0, 255, 0), 4)
    output_no += 1
    print("number of corner pixels: ", str(len(corners)))
    print("number of final corner pixels: ", str(len(final)))
    cv.imwrite(os.path.join(output_path, '%d' % output_no + '.jpg'), out)
    print("DONE ", str(output_no))


#for i in range(number_files - 1):
img2 = cv.imread(files[number_files - 1])
#print(img2.shape)
for g in range(1, number_files):
    i = number_files - g
    img1 = cv.imread(files[i - 1])
    #img2 = rgb2gray(temp2)
    u = power[i - 1]
    a = sorted(u, key = lambda x : x[1])
    #print(a)
    v = power[i]
    b = sorted(v, key = lambda x : x[1])
    #print(b)
    m1 = []
    m2 = []
    count = 0
    for j in a:
        for k in b:
            if count == 3:
                break
            if ((j[0] - k[0]) ** 2 + (j[1] - k[1]) ** 2 <= 500) and (k[0] == j[0]):
                m1.append([j[0], j[1]])
                m2.append([k[0], k[1]])
                count += 1
        if count == 3:
                break
    m1 = np.float32(m1)
    m2 = np.float32(m2)
    #print(m1)
    #print(m2)
    M = cv.getAffineTransform(m1, m2)
    Minv = cv.getAffineTransform(m2, m1)
    #print(M)
    
    A = 0
    B = img1.shape[0]
    C = img1.shape[1]
    C_dash = np.matmul(M, [0, C, 1])[1]
    E = np.matmul(Minv, [0, 0, 1])[1]
    G = E + img2.shape[1]
    G = round(G)
    C_dash = round(C_dash)
    
    #print(B)
    #print(G)
    
    stitched = np.zeros((B, G, 3), np.uint8)
    
    #print(img1.shape)
    #print(img2.shape)
    #print(stitched.shape)
    
    for j in range(G):
        if j < C:
            for k in range(B):
                stitched[k][j] = img1[k][j]
        else:
            for k in range(B):
                stitched[k][j] = img2[k][j - C + C_dash]
            
    #stitched[0 : B][A : C] = img1
    #stitched[0 : B][C : G] = img2[0 : B][C_dash : img2.shape[1]]
    cv.imwrite(os.path.join(final_path, '%d' % i + '.jpg'), stitched)
    img2 = stitched
    #ans = cv2.warpAffine(img, M, (cols, rows))
    print('DONE ', str(i))
cv.imwrite(os.path.join(final_path, 'panorama' + '.jpg'), stitched)