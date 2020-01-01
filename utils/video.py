import cv2
from skvideo.io import FFmpegWriter
from utils.detection import BodyDetector
from k_vision.visual import draw_regions
import torch
from torchvision import transforms
from k_vision.text import label_region


class VideoProcessor:
    def __init__(self, trained_model, use_gpu=False):

        self.use_gpu = use_gpu
        self.cap = None
        self.video_writer = FFmpegWriter(
            "output.mp4",
            inputdict={'-r': str(12)},
            outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-c:a': 'libvo_aacenc'})

        self.body_detector = BodyDetector(speed='normal')
        self.body_detector.load_model()

        self.fashion_classifier = torch.load(trained_model)
        if self.use_gpu:
            self.fashion_classifier.cuda()

        self.fashion_classifier.eval()
        self.label_dict = {0: "t-shirt/top", 1: "trouser", 2: "pullover", 3: "dress", 4: "coat",
                           5: "sandal", 6: "shirt", 7: "sneaker", 8: "bag", 9: "ankle boot"}
        self.data_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(),
                                                   transforms.Resize(28, 28), transforms.ToTensor()])

    def classify_region(self, region, frame):
        # run fashion classification model on each detected body region
        region_img = frame[region.top: region.bottom, region.left: region.right]
        region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
        # doing data transform for classification model
        region_img = self.data_transforms(region_img).expand(1, 1, 28, 28)
        output = self.fashion_classifier(region_img)
        predicted = output.argmax(dim=1, keepdim=True)

        return self.label_dict[predicted]

    def process(self, video):
        self.cap = cv2.VideoCapture(video)
        while self.cap.isOpened():
            _, frame = self.cap.read()
            if frame is not None:
                # doing body detection first
                body_regions = self.body_detector.process(frame)
                output_image = frame
                for b_r in body_regions:
                    # write predicted fashion class on the frame
                    output_image = label_region(output_image, text=self.classify_region(b_r, frame), region=b_r)
                # draw bounding box on the frame
                output_image = draw_regions(output_image, regions=body_regions)
                self.video_writer.writeFrame(output_image)
            else:
                break


if __name__ == "__main__":
    vp = VideoProcessor()
    vp.process(video="/home/xuehao/Downloads/fashion_testing.mp4")
