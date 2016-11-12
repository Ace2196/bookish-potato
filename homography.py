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
            homography_matrices = np.array([np.identity(3)])
            prev_frame = frame
            continue

        if i%20 == 0:
            homography_matrices = np.append(
                homography_matrices,
                [np.identity(3)],
                axis=0
            )
            prev_frame = frame
            continue

        H = Stitcher().find_homography(frame, prev_frame)

        homography_matrices = np.append(
            homography_matrices,
            [H],
            axis=0
        )

    initialized = False
    for i, frame in enumerate(video):
        stdout.write('{}\r'.format(i))
        stdout.flush()

        if initialized is False:
            initialized = True
            reduced_H = np.identity(3)
            prev_frame = frame
            continue

        if i%20 == 0:
            H = Stitcher().find_homography(frame, prev_frame)
            reduced_H = reduced_H @ H
            prev_frame = frame

        # H = reduced_H @ H
        homography_matrices[i] = reduced_H @ homography_matrices[i]

    return homography_matrices
