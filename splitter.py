from glob import iglob
from os import mkdir
from os.path import (
    basename,
    join,
)

import cv2

from video import Video


def split(output_dirname, video=None):
    for i, frame in enumerate(video):
        filename = '{}.png'.format(i)
        output_pathname = join(output_dirname, filename)
        cv2.imwrite(output_pathname, frame)


if __name__ == '__main__':
    video_pathname_pattern = 'beachVolleyball/*.mov'
    output_dirname = 'images'
    mkdir(output_dirname)

    for video_pathname in iglob(video_pathname_pattern):
        filename = basename(video_pathname)
        video_dirname = join(output_dirname, filename)
        mkdir(video_dirname)

        video = Video(video_pathname)

        split(video_dirname, video)
