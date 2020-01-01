import cv2
from skvideo.io import FFmpegWriter
from utils.detection import BodyDetector
from k_vision.visual import draw_regions
from k_vision.text import label_region


class VideoProcessor:
    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu
        self.cap = None
        self.video_writer = FFmpegWriter(
            "output.mp4",
            inputdict={'-r': str(12)},
            outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-c:a': 'libvo_aacenc'})

        self.body_detector = BodyDetector(speed='normal')
        self.body_detector.load_model()


    def process(self, video):

        self.cap = cv2.VideoCapture(video)

        while self.cap.isOpened():

            _, frame = self.cap.read()

            if frame is not None:

                body_regions = self.body_detector.process(frame)

                output_image = draw_regions(frame, regions=body_regions)

                self.video_writer.writeFrame(output_image)

            else:
                break


if __name__ == "__main__":
    vp = VideoProcessor()
    vp.process(video="/home/xuehao/Downloads/fashion_testing.mp4")

