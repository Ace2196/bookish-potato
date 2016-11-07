import cv2


class Video(object):
    class CV2Video(object):
        propIds = (
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
        )

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
            else:
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

if __name__ == '__main__':
    from pprint import pprint
    video = Video('beachVolleyball/beachVolleyball1.mov')
    pprint(video.props)

    import numpy as np
    pprint(np.array_equal(video, video))
