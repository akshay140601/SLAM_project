'''
TODO: Checkout flann based matcher instead of brute force
'''

import cv2

class feat_det_mat():
    def __init__(self, method='ORB') -> None:
        self.method = method
        
    def sift_detector_and_match(self, imgl, imgr):
        imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(imgl, None)
        kp2, desc2 = sift.detectAndCompute(imgr, None)
        #print(desc1)
        #print(desc2)

        # Matching
        # Using L1 distance for computation time. Change to L2 if performance is good
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(desc1, desc2)
        # Sorting them based on distances
        matches = sorted(matches, key = lambda x:x.distance)
        # Selecting 75% of best matches
        num_good_matches = int(0.25 * len(matches))
        # Select using Lowe's test
        '''good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])'''

        img_match = cv2.drawMatches(imgl, kp1, imgr, kp2, matches[:num_good_matches], imgr, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        return kp1, kp2, matches, img_match


    def orb_detector_and_match(self, imgl, imgr):
        imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp1, desc1 = orb.detectAndCompute(imgl, None)
        kp2, desc2 = orb.detectAndCompute(imgr, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        '''good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])'''
        
        img_match = cv2.drawMatches(imgl, kp1, imgr, kp2, matches, imgr, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return kp1, kp2, matches, img_match

    def detect_and_match(self, imgl, imgr):
        if self.method == 'ORB':
            return self.orb_detector_and_match(imgl, imgr)
        else:
            return self.sift_detector_and_match(imgl, imgr)