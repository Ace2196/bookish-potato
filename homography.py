import numpy as np
from sys import stdout

from stitcher import Stitcher


def homography_matrices(video):
    initialized = False
    for i, frame in enumerate(video):
        stdout.write('{}\r'.format(i))
        stdout.flush()

        if initialized is False:
            initialized = True
            reduced_H = np.identity(3)
            homography_matrices = np.array([reduced_H])
            prev_frame = frame
            continue

        H = Stitcher().find_homography(frame, prev_frame)

        reduced_H = reduced_H @ H
        homography_matrices = np.append(
            homography_matrices,
            [reduced_H],
            axis=0
        )

        prev_frame = frame

    return homography_matrices
