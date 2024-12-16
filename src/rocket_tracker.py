import cv2
import numpy as np
from inference import InferencePipeline
import supervision as sv
import RPi.GPIO as GPIO
import time

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

VERSION = 1  # Set your model version here
NMS_MAX_OVERLAP = 1.0
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = None
CONFIDENCE = 0.20
OVERLAP = 0.30
IMG_SIZE = 640
SOURCE = '/home/pi/Videos/video_2.mp4'  # Default source




class PanTiltController:
    def __init__(self, 
                 pan_pin=17,    # PWM pin for pan servo
                 tilt_pin=27,   # PWM pin for tilt servo
                 min_pan=2.5,   # Minimum pulse width for pan servo
                 max_pan=12.5,  # Maximum pulse width for pan servo
                 min_tilt=2.5,  # Minimum pulse width for tilt servo
                 max_tilt=12.5  # Maximum pulse width for tilt servo
                ):
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pan_pin, GPIO.OUT)
        GPIO.setup(tilt_pin, GPIO.OUT)
        
        # Create PWM instances
        self.pan_servo = GPIO.PWM(pan_pin, 50)  # 50 Hz (20ms PWM period)
        self.tilt_servo = GPIO.PWM(tilt_pin, 50)
        
        # Start PWM
        self.pan_servo.start(0)
        self.tilt_servo.start(0)
        
        # Store servo parameters
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.min_pan = min_pan
        self.max_pan = max_pan
        self.min_tilt = min_tilt
        self.max_tilt = max_tilt
        
        # Current positions
        self.current_pan = 7.5  # Neutral position
        self.current_tilt = 7.5  # Neutral position

    def set_pan_angle(self, angle):
        """
        Set pan servo to specific angle
        angle: -90 to 90 degrees
        """
        # Constrain angle
        angle = max(-90, min(90, angle))
        
        # Map angle to pulse width
        pulse = self.min_pan + (angle + 90) * (self.max_pan - self.min_pan) / 180
        
        self.pan_servo.ChangeDutyCycle(pulse)
        self.current_pan = pulse
        time.sleep(0.3)  # Allow servo time to move

    def set_tilt_angle(self, angle):
        """
        Set tilt servo to specific angle
        angle: -90 to 90 degrees
        """
        # Constrain angle
        angle = max(-90, min(90, angle))
        
        # Map angle to pulse width
        pulse = self.min_tilt + (angle + 90) * (self.max_tilt - self.min_tilt) / 180
        
        self.tilt_servo.ChangeDutyCycle(pulse)
        self.current_tilt = pulse
        time.sleep(0.3)  # Allow servo time to move

    def track_object(self, detection, frame_width, frame_height):
        """
        Track an object by adjusting pan and tilt based on object's position
        
        :param detection: Single object detection dictionary
        :param frame_width: Width of the video frame
        :param frame_height: Height of the video frame
        """
        # Calculate object's center
        x = detection['x'] + detection['width'] / 2
        y = detection['y'] + detection['height'] / 2
        
        # Calculate offset from frame center
        x_offset = (x / frame_width - 0.5) * 2  # Normalize to -1 to 1
        y_offset = (y / frame_height - 0.5) * 2  # Normalize to -1 to 1
        
        # Convert offsets to servo angles
        pan_angle = x_offset * 45  # Max 45 degrees left/right
        tilt_angle = -y_offset * 45  # Max 45 degrees up/down (inverted due to image coordinates)
        
        # Adjust servos
        self.set_pan_angle(pan_angle)
        self.set_tilt_angle(tilt_angle)

    def cleanup(self):
        """
        Reset servos and cleanup GPIO
        """
        self.pan_servo.stop()
        self.tilt_servo.stop()
        GPIO.cleanup()


class ObjectTracker:
    def __init__(self, 
                 api_key, 
                 model_id, 
                 video_source, 
                 confidence_threshold=0.40,
                 max_cosine_distance=0.4):
        
        self.api_key = api_key
        self.model_id = model_id
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold

        max_cosine_distance = MAX_COSINE_DISTANCE
        
        # Tracker configuration
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", 
            max_cosine_distance, 
            None  # No budget limit
        )
        self.tracker = Tracker(metric)

        self.pan_tilt = PanTiltController()
        
        # Annotators
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        
    def _extract_features(self, image, bbox):  
        """
        Extracts features from an image patch defined by bbox.
        """
        x1, y1, width, height = map(int, bbox)
        x2, y2 = x1 + width, y1 + height
        
         # Ensure bbox is within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        # Extract patch
        patch = image[y1:y2, x1:x2]

        # Resize to a fixed size (e.g., 64x64) and flatten
        if patch.size > 0:
            patch_resized = np.resize(patch, (64, 64))
            patch_resized = patch_resized / 255.0

            feature = patch_resized.flatten().astype(np.float32)
        else:
            # Default feature if patch is empty
            feature = np.zeros(64 * 64 * 3, dtype=np.float32)

            return feature

    def custom_sink(self, predictions, video_frame):
        """
        Process predictions and update object tracking
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
            video_frame = [video_frame]
        
        for prediction, frame in zip(predictions, video_frame):
            if prediction is None:
                continue
            
            # Filter predictions by confidence
            filtered_predictions = [
                pred for pred in prediction.get("predictions", []) 
                if pred["confidence"] >= self.confidence_threshold
            ]
            
            # Convert predictions to detection format
            detections = []
            for pred in filtered_predictions:
                    # Convert class to integer if it's not already
                    class_id = pred.get('class_id', 0)
                    if isinstance(class_id, str):
                        try:
                            class_id = int(class_id)
                        except ValueError:
                            class_id = 0  # default to 0 if conversion fails
                    
                    # Extract bounding box
                    bbox = [
                        pred["x"] - pred["width"] / 2, 
                        pred["y"] - pred["height"] / 2, 
                        pred["width"], 
                        pred["height"]
                    ]
                    
                    # Feature extraction 
                    # feature = self._extract_features(frame.image, bbox)
                    
                    detection = Detection(
                        tlwh=bbox,
                        confidence=pred["confidence"],
                        class_num=class_id,
                        feature=None  # Add feature extraction
                    )
                    detections.append(detection)
                
                # Non-maximum suppression
            if detections:
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    class_nums = np.array([d.class_num for d in detections])
                    
                    indices = preprocessing.non_max_suppression(
                        boxes, class_nums, 1.0, scores
                    )
                    detections = [detections[i] for i in indices]
                

            
            # Update tracker
            self.tracker.predict()
            self.tracker.update(detections)

            if detections:
                # Sort by confidence and select the top detection
                top_detection = max(filtered_predictions, key=lambda x: x['confidence'])

                # Track the object with pan-tilt mechanism
                self.pan_tilt.track_object(
                    top_detection,
                    frame_width=frame.image.shape[1],
                    frame_height=frame.image.shape[0]
                )
            
            # Prepare labels (ensure they are strings)
            labels = [str(d.class_num) for d in detections]
            
            # Annotate frame with tracked objects
            image = self.annotate_frame(frame.image, detections, labels)
            
            # Resize and display
            resized_image = cv2.resize(image, (1280,720))
            cv2.imshow("Object Tracking", resized_image)
            cv2.waitKey(1)

    def annotate_frame(self, image, detections, labels):
        """
        Annotate frame with bounding boxes and labels
        """
        # Create Supervision Detections object
        detection_sv = sv.Detections(
            xyxy=np.array([d.to_tlbr() for d in detections]),
            confidence=np.array([d.confidence for d in detections]),
            class_id=np.array([d.class_num for d in detections])
        )

        # Annotate with box and label
        annotated_image = self.box_annotator.annotate(
            scene=image.copy(), 
            detections=detection_sv
        )
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, 
            detections=detection_sv, 
            labels=labels
        )
        
        return annotated_image
    

    def run(self):
        """
        Initialize and start inference pipeline
        """
        pipeline = InferencePipeline.init(
            model_id=self.model_id,
            video_reference=self.video_source,
            on_prediction=self.custom_sink,
            api_key=self.api_key
        )
        
        try:
            pipeline.start()
            pipeline.join()
        except KeyboardInterrupt:
            print("Tracking stopped by user.")
        finally:
            # Ensure GPIO is cleaned up
            self.pan_tilt.cleanup()
            cv2.destroyAllWindows()

def main():
    API_KEY = ""
    MODEL_ID = ""
    VIDEO_SOURCE = SOURCE
    


    tracker = ObjectTracker(
        api_key=API_KEY, 
        model_id=MODEL_ID, 
        video_source=VIDEO_SOURCE
    )
    tracker.run()

if __name__ == "__main__":
    main()
