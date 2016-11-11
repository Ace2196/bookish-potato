from itertools import compress
from math import ceil
from operator import (
    attrgetter,
    itemgetter,
)

import cv2
import numpy as np

from utils import(
    isLine,
    isSand,
)


class Stitcher(object):
    def find_homography(
        self,
        src,
        dst,
        ratio=.75,
        ransacReprojThreshold=4
    ):
        kp_des_src, kp_des_dst = map(
            self.detect_and_describe,
            (src, dst)
        )

        H = self._find_homography(
            kp_des_src,
            kp_des_dst,
            ratio,
            ransacReprojThreshold
        )

        return H

    def detect_and_describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = cv2.xfeatures2d.SIFT_create()
        kps = detector.detect(gray)

        court_kps = []
        for kp in kps:
            x, y = map(
                lambda c: int(ceil(c)),
                kp.pt
            )

            rgb = image[y, x][::-1]

            if (
                isSand(*rgb) or
                isLine(*rgb)
            ):
                court_kps.append(kp)

        kps, des = detector.compute(gray, court_kps)

        kps = np.float64([kp.pt for kp in kps])

        return kps, des

    def _find_homography(
        self,
        kp_des_src,
        kp_des_dst,
        ratio,
        ransacReprojThreshold
    ):
        kp_src, des_src = kp_des_src
        kp_dst, des_dst = kp_des_dst

        matcher = cv2.DescriptorMatcher_create('BruteForce')
        raw_matches = matcher.knnMatch(des_src, des_dst, 2)

        good_matches = filter(
            lambda match: (
                len(match) == 2 and
                match[0].distance < match[1].distance * ratio
            ),
            raw_matches
        )

        matches_src = list(map(itemgetter(0), good_matches))

        indices_src = map(
            attrgetter('queryIdx'),
            matches_src
        )

        indices_dst = map(
            attrgetter('trainIdx'),
            matches_src
        )

        pts_src = np.float64([kp_src[i] for i in indices_src])
        pts_dst = np.float64([kp_dst[i] for i in indices_dst])

        H, mask = cv2.findHomography(
            pts_src,
            pts_dst,
            cv2.RANSAC,
            ransacReprojThreshold
        )

        return H

    def warp(self, src, H):
        warped_image = cv2.warpPerspective(
            src,
            H,
            src.shape[1::-1]
        )

        return warped_image

if __name__ == '__main__':
    src = cv2.imread('images/beachVolleyball1.mov/404.png')
    dst = cv2.imread('images/beachVolleyball1.mov/000.png')

    stitcher = Stitcher()
    stitched_image, H = stitcher.stitch(src, dst)

    cv2.imwrite('asdf.png', stitched_image)
