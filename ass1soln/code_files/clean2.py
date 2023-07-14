import numpy as np
import cv2 as cv

b = int(shape[0] / 2)
l = int(shape[1] / 7)
print(l, b)
#area = l * b

def IoU(x1, y1, x2, y2):
    a1 = x1 + b
    b1 = y1 + l
    a2 = x2 + b
    b2 = y2 + l
    xx = max(x1, x2)
    yy = max(y1, y2)
    aa = min(a1, a2)
    bb = min(b1, b2)
    
    w = max(0, aa - xx)
    h = max(0, bb - yy)
    return (w * h) / (2 * area - w * h)
    

# Draw parameters
thresh = 0.1


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
    out = img1.copy()
                             
    for i in range(1, shape[0]):
        integral[i][0] += integral[i - 1][0]
    for j in range(1, shape[1]):
        integral[0][j] += integral[0][j - 1]
                             
    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            integral[i][j] += integral[i - 1][j] + integral[i][j - 1] - integral[i - 1][j - 1]
            
    list_my = []
    scores = []
    
    for i in range(shape[0] - b):
        for j in range(shape[1] - l):
            score = integral[i + b][j + l] - integral[i][j + l] - integral[i + b][j] + integral[i][j]
            score = score / 255
            if score / area <= 0.95:
                list_my.append([i, j])
                scores.append(score)

    scores = np.array(scores)
    order = scores.argsort()
    keep = []
    
    while(len(order) > 0):
        index = order[-1]
        keep.append(list_my[index])
        P = list_my[index]
        order = order[: -1]
        if len(order) == 0:
            break
        mask = []
        for i in range(len(order)):
            box = list_my[order[i]]
            if IoU(P[0], P[1], box[0], box[1]) < 0.1:
                mask.append(True)
            else:
                mask.append(False)
        order = order[mask]
                
    for el in keep:
        cv.rectangle(out, (int(el[1]), int(el[0])), (int(el[1] + l), int(el[0] + b)), (255, 0, 0), 1)
        
    cv.imwrite(os.path.join(path, '%06d' % file_no +'.png'), out)
    print(file_no)
    file_no += 1
    