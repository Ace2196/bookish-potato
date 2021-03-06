from glob import iglob
from os.path import join

import cv2
import numpy as np


def from_images(pathname_pattern):
    pathnames = sorted(iglob(pathname_pattern))

    frames = np.array(list(map(cv2.imread, pathnames)))

    return frames


def write_images(it, output_dirname, num_frames=None):
    if not num_frames:
        try:
            num_frames = len(it)
        except TypeError:
            it = list(it)
            num_frames = len(it)

    last_index = num_frames - 1
    length = len(str(last_index))

    for i, frame in enumerate(it):
        filename = '{}.png'.format(str(i).zfill(length))

        output_pathname = join(output_dirname, filename)

        cv2.imwrite(output_pathname, frame)


class Video(object):
    class CV2Video(object):
        propIds = {
            cv2.CAP_PROP_POS_MSEC,
            cv2.CAP_PROP_POS_FRAMES,
            cv2.CAP_PROP_POS_AVI_RATIO,
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FOURCC,
            cv2.CAP_PROP_FRAME_COUNT,
            cv2.CAP_PROP_FORMAT,
            cv2.CAP_PROP_MODE,
            cv2.CAP_PROP_BRIGHTNESS,
            cv2.CAP_PROP_CONTRAST,
            cv2.CAP_PROP_SATURATION,
            cv2.CAP_PROP_HUE,
            cv2.CAP_PROP_GAIN,
            cv2.CAP_PROP_EXPOSURE,
            cv2.CAP_PROP_CONVERT_RGB,
            cv2.CAP_PROP_RECTIFICATION,
            cv2.CAP_PROP_ISO_SPEED,
            cv2.CAP_PROP_BUFFERSIZE,
        }

        def __init__(self, pathname):
            self.pathname = pathname

        def __enter__(self):
            self.cap = cv2.VideoCapture(self.pathname)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.cap.release()

        def __iter__(self):
            return self

        def __next__(self):
            ret, frame = self.cap.read()

            if ret:
                return frame

            raise StopIteration

        def props(self):
            return {
                propId: self.cap.get(propId)
                for propId in self.propIds
            }

    def __init__(self, pathname):
        self.pathname = pathname
        self.set_props()

    def __iter__(self):
        with self.open() as video:
            for frame in video:
                yield frame

    def open(self):
        return self.CV2Video(self.pathname)

    def set_props(self):
        with self.open() as video:
            self.props = video.props()

    def prop(self, propId):
        return self.props[propId]

    def as_array(self):
        return np.array(list(self.__iter__()))

    def write_images(self, output_dirname):
        write_images(self.__iter__(), output_dirname)
