import numpy as np
import cv2 as cv
import os
import skimage
from skimage import io, util
from scipy.stats import norm


data_path = os.path.join(os.path.dirname(os.path.abspath('Assignment1.ipynb')), 'COL780 Dataset/Candela_m1.10/input')
output_path = os.path.join(os.path.dirname(os.path.abspath('Assignment1.ipynb')), 'COL780 Dataset/Candela_m1.10/result')

number_files = len(os.listdir(data_path))
files = []
for i in range(number_files):
    file_name = os.path.join(data_path, 'Candela_m1.10_' + '%06d' % i + '.png')
    files.append(file_name)
img1 = cv.imread(files[0])
shape = img1.shape
gray1 = np.array([[0 for j in range(shape[1])] for i in range(shape[0])])
print(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        gray1[i][j] = 0.2989*img1[i][j][0] + 0.5870*img1[i][j][1] + 0.1140*img1[i][j][2]


K = 4
Tb = 0.8
alpha = 0.5
weight = [0.6, 0.25, 0.1, 0.05]
sigma_s = 225.0

height = shape[0] 
width = shape[1]

class bgd_sub:
    
    def __init__(self):
        self.wt = None
        self.mu = None
        self.sigma_sq = None
        self.result = None
        
    def match(self, pixel, mu, sigma):
        if sigma == 0:
            sigma = 0.01
        return np.absolute((pixel - mu) / np.sqrt(sigma)) <= 3
    
    def main(self):
        self.wt = np.array([[weight for j in range(shape[1])] for i in range(shape[0])])
        self.mu = np.array([[[gray1[i][j] for k in range(K)] for j in range(shape[1])] for i in range(shape[0])])
        self.sigma_sq = np.array([[[sigma_s for k in range(K)] for j in range(shape[1])] for i in range(shape[0])])
        self.result = np.array([[0 for j in range(shape[1])] for i in range(shape[0])])
        
        output_no = 0
        for file in files:
            img = cv.imread(file)
            gray = np.array([[0 for j in range(shape[1])] for i in range(shape[0])])

            for i in range(shape[0]):
                for j in range(shape[1]):
                    gray[i][j] = 0.2989*img[i][j][0] + 0.5870*img[i][j][1] + 0.1140*img[i][j][2]

            for i in range(shape[0]):
                for j in range(shape[1]):
                    matched_gaussian = -1
                    for k in range(K):
                        if self.match(gray[i][j], self.mu[i][j][k], self.sigma_sq[i][j][k]):
                            matched_gaussian = k
                            break
                    if matched_gaussian == -1:
                        temp = [self.wt[i][j][k] for k in range(K)]
                        lowest_wt_gaussian = temp.index(min(temp))
                        self.mu[i][j][lowest_wt_gaussian] = gray[i][j]
                        self.sigma_sq[i][j][lowest_wt_gaussian] = sigma_s
                    else:
                        for k in range(K):
                            self.wt[i][j][k] = (1 - alpha) * self.wt[i][j][k]
                        self.wt[i][j][matched_gaussian] += alpha
                        
                        matched_mu = self.mu[i][j][matched_gaussian]
                        matched_sigma_sq = self.sigma_sq[i][j][matched_gaussian]
                        
                        if matched_sigma_sq == 0:
                            matched_sigma_sq = 0.1
                            
                        pdf_val = norm.pdf(gray[i][j] , loc = matched_mu , scale = np.sqrt(matched_sigma_sq))
                        rho = alpha * pdf_val
                        
                        self.mu[i][j][matched_gaussian] = int((1 - rho) * matched_mu + rho * gray[i][j])
                        diff = gray[i][j] - self.mu[i][j][matched_gaussian]
                        diff = diff ** 2
                        self.sigma_sq[i][j][matched_gaussian] = (1 - rho) * matched_sigma_sq + rho * diff
                    
                    for k in range(K):
                        if self.sigma_sq[i][j][k] == 0:
                            self.sigma_sq[i][j][k] = 0.01

                    ratio = self.wt[i][j] / np.sqrt(self.sigma_sq[i][j])
                    order = np.argsort(ratio)
                    order = np.flip(order)
                    total_bkg_wt = 0
                    index = 0
                    while(total_bkg_wt <= Tb):
                        total_bkg_wt += self.wt[i][j][order[index]]
                        index += 1
                                        
                    for t in range(index):
                        if self.match(gray[i][j], self.mu[i][j][order[t]], self.sigma_sq[i][j][order[t]]):
                            self.result[i][j] = 255
                        else:
                            self.result[i][j] = 0
                            break
                            
            cv.imwrite(os.path.join(output_path, '%06d' % output_no +'.png'), self.result)
            print("DONE ", str(output_no))
            output_no += 1

model = bgd_sub()
model.main()
