import numpy as np
from sys import stdout

from stitcher import Stitcher
from video import (
    Video,
    write_images,
)


if __name__ == '__main__':
    stitcher = Stitcher()
    video = Video('beachVolleyball/beachVolleyball1.mov')

    initialized = False
    for i, frame in enumerate(video):
        stdout.write('{}\r'.format(i))
        stdout.flush()

        if initialized is False:
            initialized = True
            reduced_H = np.identity(3)
            warped_images = np.array([frame])
            homography_matrices = np.array([reduced_H])
            prev_frame = frame
            continue

        H = stitcher.find_homography(frame, prev_frame)

        reduced_H = reduced_H @ H
        homography_matrices = np.append(
            homography_matrices,
            [reduced_H],
            axis=0
        )

        warped_image = stitcher.warp(frame, reduced_H)
        warped_images = np.append(
            warped_images,
            [warped_image],
            axis=0
        )

        prev_frame = frame

    write_images(warped_images, 'test')
