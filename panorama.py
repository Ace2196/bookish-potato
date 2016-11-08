import itertools
import sys

import cv2
import numpy as np


detector = cv2.xfeatures2d.SIFT_create()


def detect(gray):
    return detector.detectAndCompute(gray, None)

matcher = cv2.BFMatcher()


def match(des1, des2, ratio=.75):
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    matches1, matches2 = zip(
        *(
            (m1.queryIdx, m1.trainIdx)
            for m1, m2 in raw_matches
            if m1.distance < m2.distance * ratio
        )
    )

    return matches1, matches2


class Stitcher(object):
    def __init__(self, detect=detect, match=match):
        self.detect = detect
        self.match = match

    def panorama(self, video):
        panorama0 = None

        for i, frame in enumerate(video):
            sys.stdout.write('\r{}'.format(i))
            sys.stdout.flush()

            if panorama0 is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp0, des0 = self.detect(gray)
                start0 = 0
                start1 = 0
                H = np.identity(3)
                height, width = frame.shape[:2]
                start0 = 0
                start1 = 0
                frame_coor0 = np.array(
                    [
                        [(i0, i1) for i1 in range(width)]
                        for i0 in range(height)
                    ],
                    np.float64
                )
                frame_count0 = np.ones(frame.shape)
                panorama0 = frame
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kp1, des1 = self.detect(gray)

            matches0, matches1 = map(
                np.array,
                self.match(des0, des1)
            )

            pts0 = np.float64([kp.pt for kp in kp0])[matches0]
            pts1 = np.float64([kp.pt for kp in kp1])[matches1]

            M, mask = cv2.findHomography(
                pts1,
                pts0,
                cv2.RANSAC,
                ransacReprojThreshold=4
            )

            H = H @ M

            # XXX upsample/downsample to better approximate float coordinates?
            corners = np.float64(
                [
                    [(i0, i1) for i1 in (0, width - 1)]
                    for i0 in (0, height - 1)
                ]
            )

            corners = np.rint(cv2.perspectiveTransform(corners, H)).astype(int)

            # Resize panorama/frame_count
            min0 = corners[:, :, 0].min()
            max0 = corners[:, :, 0].max()

            min1 = corners[:, :, 1].min()
            max1 = corners[:, :, 1].max()

            # Update start indices and shape
            offset0 = -min(0, start0 + min0)
            start0 += offset0

            offset1 = -min(0, start1 + min1)
            start1 += offset1

            # Update shape
            shape = (
                max(
                    panorama0.shape[0] + offset0,
                    start0 + height,
                    start0 + max0 + 1
                ),
                max(
                    panorama0.shape[1] + offset1,
                    start1 + width,
                    start1 + max1 + 1
                ),
                3
            )

            panorama1 = np.zeros(shape)
            panorama1[
                offset0:offset0 + panorama0.shape[0],
                offset1:offset1 + panorama0.shape[1]
            ] = panorama0

            frame_count1 = np.zeros(shape)
            frame_count1[
                offset0:offset0 + panorama0.shape[0],
                offset1:offset1 + panorama0.shape[1]
            ] = frame_count0

            # Transform frame coordinates
            frame_coor1 = np.rint(
                cv2.perspectiveTransform(frame_coor0, H)
            ).astype(int)

            # XXX Too tired to figure out array operations
            for i0, i1 in itertools.product(
                range(height),
                range(width)
            ):
                new_i0, new_i1 = frame_coor1[i0, i1]
                new_i0 += start0
                new_i1 += start1

                frame_count1[new_i0, new_i1] += 1
                alpha = 1/frame_count1[new_i0, new_i1]

                panorama1[new_i0, new_i1] *= 1 - alpha
                panorama1[new_i0, new_i1] += frame[i0, i1] * alpha

            panorama0 = panorama1
            frame_count0 = frame_count1

            cv2.imwrite(
                'images/panorama{}.jpg'.format(i),
                panorama0
            )

        return panorama0
