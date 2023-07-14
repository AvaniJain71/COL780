import numpy as np
import cv2 as cv

s = shape[0] / 15
t = 0.1
output = [] 
for i in range(number_files):
    file_name = os.path.join(output_path, '%06d' % i + '.png')
    output.append(file_name)
path = os.path.join(os.path.dirname(output_path), 'final/')

file_no = 0                    
for file in output:
    img1 = cv.imread(file)
    img = img1[:, :, 0]
    integral = np.array([[int(img[i][j]) for j in range(shape[1])] for i in range(shape[0])])
    
    out = np.array([[255 for j in range(shape[1])] for i in range(shape[0])])
                             
    for i in range(1, shape[0]):
        integral[i][0] += integral[i - 1][0]
    for j in range(1, shape[1]):
        integral[0][j] += integral[0][j - 1]
                             
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            integral[i][j] += integral[i - 1][j] + integral[i][j - 1] - integral[i - 1][j - 1]
                             
    for i in range(shape[0]):
        for j in range(shape[1]):
            x1 = int(max(i - s/2, 0))
            x2 = int(min(i + s/2, shape[0] - 1))
            y1 = int(max(j - s/2, 0))
            y2 = int(min(j + s/2, shape[1] - 1))
            count = (x2 - x1) * (y2 - y1)
            sum_my = integral[x2][y2] + integral[x1][y1] - integral[x1][y2] - integral[x2][y1]
            sum_my /= 255
            if sum_my == 0:
                sum_my = 0.1
            if (count - sum_my) / count >= t:
                for x in range(x1 + 1, x2 + 1):
                    for y in range(y1 + 1, y2 + 1):
                        out[x][y] = 0
    cv.imwrite(os.path.join(path, '%06d' % file_no +'.png'), out)
    print(file_no)
    file_no += 1
    