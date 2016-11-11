from glob import iglob
from os import mkdir
from os.path import (
    basename,
    join,
)

import numpy as np
from sys import stdout

from stitcher import Stitcher
from video import Video


def homography_matrices(video):
    stitcher = Stitcher()

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


if __name__ == '__main__':
    video_pathname_pattern = 'beachVolleyball/*.mov'
    matrices_dirname = 'homography_matrices'

    video_pathnames = iglob(video_pathname_pattern)
    mkdir(matrices_dirname)

    for video_pathname in video_pathnames:
        video = Video(video_pathname)

        video_filename = basename(video_pathname)
        matrices_pathname = join(matrices_dirname, video_filename + '.txt')

        matrices = homography_matrices(video)

        np.savetxt(matrices_pathname, matrices)
