import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

row = 0
column = 0
K = 0

def initialize():

    global row, column, K, board
    board = np.zeros(shape=(row, column, K), dtype=object)
    for i in xrange(row):
        for j in xrange(column):
            for k in xrange(K):
                board[i][j][k] = {'mean': np.array([0, 0, 0]), 'deviation': 1, 'weight': 0.33}

def update(pixel, data, distance, alpha, rho):
    if distance < 2.5 * pixel['deviation']:
        pixel['weight'] = (1 - alpha) * pixel['weight'] + alpha
        pixel['mean'] = (1 - rho) * pixel['mean'] + rho * data
        pixel['deviation'] = np.sqrt((1 - rho) * (pixel['deviation']**2) + rho * (np.dot(np.subtract(data, pixel['mean']), np.subtract(data, pixel['mean']).T)))
        return True

    else:
        pixel['weight'] = (1 - alpha) * pixel['weight']
        return False

def mahalanobis(pixel, data):

    temp1 = np.subtract(data, pixel['mean'])
    temp2 = np.dot(temp1, np.linalg.inv((pixel['deviation']**2) * np.eye(3)))
    distance = math.sqrt(np.dot(temp2, temp1.T))
    return distance

def probability(pixel, data):

    global K

    temp1 = np.subtract(data, pixel['mean'])
    temp2 = np.dot(temp1, np.linalg.inv((pixel['deviation']**2) * np.eye(3)))
    power = -1 * (np.dot(temp2, temp1.T) / 2.0)

    if power > 10:
        return 0.0000001
    # print power, pixel['mean']
    prob = np.exp(power)
    prob /= math.sqrt(math.pow(2*math.pi, 3) * np.linalg.det((pixel['deviation']**2) * np.eye(3)))
    
    # print prob
    return prob

def check(pixel):

    global K
    ratio = []
    idx = [0, 1, 2]
    for i in xrange(K):
        ratio.append(pixel[i]['weight']/pixel[i]['deviation'])

    temp = zip(ratio, idx)
    temp.sort()
    ratio, idx = zip(*temp)
    ans = [-1, -1, -1]
    curr = 0
    T = 0.3
    # print ratio
    for i in xrange(K):
        curr += ratio[i]
        if curr > T:
            ans[idx[i]] = 0         # Foreground
        
        else:
            ans[idx[i]] = 1         # Background
    
    return ans

ans = []
def frame_processing(video):

    global board, K, ans

    for i in xrange(row):
        for j in xrange(column):
            data = np.array(video[i][j])
            matched = False
            mini = 2
            divide = check(board[i][j])

            for k in xrange(K):
                alpha = 0.5
                prob = probability(board[i][j][k], data)
                # prob = 0.03
                if prob < mini:
                    least = k

                rho = alpha * prob
                distance = mahalanobis(board[i][j][k], data)
                verdict = update(board[i][j][k], data, distance, alpha, rho)
                if verdict is True:                  
                    if divide[k] == 1:
                        video[i][j] = [255, 255, 255]

                    else:
                        video[i][j] = [0, 0, 0]

                matched = matched or verdict

            if matched is False:                     
                video[i][j] = [0, 0, 0]
                board[i][j][least]['mean'] = data
                board[i][j][least]['deviation'] = 10
                # board[i][j][least]['weight'] = 0.3

    # print frame
    cv2.imshow('frame', video)
    ans.append(video)

def main():

    global row, column, K, ans
    cap = cv2.VideoCapture('test.avi')
    # cap = cv2.VideoCapture('test1.mp4')

    K = 3
    row = 200
    column = 320
    initialize()
    while(1):
        ret, frame = cap.read()
        # frame = cv2.split(frame)
        # print len(frame), len(frame[0])
        frame_processing(frame)
        # cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            # temp = fgmask
            break

    cap.release()
    cv2.destroyAllWindows()
    t = int(raw_input("sadf"))
    for i in xrange(len(ans)):
        cv2.imshow('frame', ans[i])

main()
