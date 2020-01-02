from imageai.Detection import ObjectDetection
from k_util.region import Region
import os
import time


class BodyDetector:
    def __init__(self, speed='normal'):
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.load_model()
        self.detector.loadModel(detection_speed=speed)
        self.custom_objects = self.detector.CustomObjects(person=True)

    def load_model(self, model_path='../downloaded'):
        self.detector.setModelPath(os.path.join(model_path, 'resnet50_coco_best_v2.0.1.h5'))

    def process(self, image):
        latest_frame = image.copy()
        regions = []
        tmp_t = time.time()

        # body detection, set min confidence to 0.7
        detected_image_array, detections = self.detector.detectCustomObjectsFromImage(
            custom_objects=self.custom_objects,
            input_type="array",
            input_image=latest_frame,
            output_type="array",
            minimum_percentage_probability=70)

        print(f'image detection time. {time.time() - tmp_t}')

        for person in detections:

            # wrap detected results to Region for following process
            region = Region()
            region.confidence = float(person['percentage_probability'])
            region.set_rect(left=person['box_points'][0],
                            right=person['box_points'][2],
                            top=person['box_points'][1],
                            bottom=person['box_points'][3])

            regions.append(region)

        return regions


