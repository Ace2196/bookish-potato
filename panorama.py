import numpy as np
import cv2
import imutils
from math import ceil, fabs

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
    
    def stitch(self, cap, count=5, ratio=0.75, reprojThresh=4.0):
        frameCount = int(cap.get(7))
        frameIterativeLength = int(frameCount/count)
        
        _, result = cap.read()
        for i in range(1, count):
            # Jump to next frame iterative length
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frameIterativeLength)
            
            _, nxt = cap.read()
            (kpsA, featuresA) = self.detectAndDescribe(result)
            (kpsB, featuresB) = self.detectAndDescribe(nxt)

            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,
                featuresA, featuresB, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                print('no match')
                continue

            # otherwise, apply a perspective warp to stitch the images
            # together
            (matches, H, status) = M
            
            movement = self.detectDirection(kpsA, kpsB, matches, status)
            
            if movement[0] < 0:
                # Movement is from right to left
                result = self.stitchImages(result, nxt, ratio, reprojThresh)
            else:
                # Movement is from left to right
                result = self.stitchImages(nxt, result, ratio, reprojThresh)
                
        return result
            
    
    def stitchImages(self, right, left, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (kpsA, featuresA) = self.detectAndDescribe(right)
        (kpsB, featuresB) = self.detectAndDescribe(left)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        
        result = cv2.warpPerspective(right, H,
            (right.shape[1] + left.shape[1], right.shape[0]))
        result[0:left.shape[0], 0:left.shape[1]] = left

        # return the stitched image
        return result
    
    def removeBlankSpace(self, image):
        i = 0
        col = result[:, i]
        while sum(sum(col)) != 0 and i < image.shape[1]:
            i += 1
            col = result[:, i]
        
        return image[:, :i]

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
    
    def detectDirection(self, kpsA, kpsB, matches, status):
        movement = np.zeros(2)
        
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = np.array((int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1])))
                ptB = np.array((int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1])))
                diff = ptA - ptB
                movement += diff
                
        return movement / sum(status)[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture('./beachVolleyball/beachVolleyball7.mov')
    s = Stitcher()
    result = s.stitch(cap, count=5)
    result = s.removeBlankSpace(result)

    plt.figure('result')
    plt.imshow(result)