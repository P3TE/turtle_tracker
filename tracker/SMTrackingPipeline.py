#!/usr/bin/env python3

import os
import sys
import csv
import yaml
from typing import List, Dict, Tuple, Optional
from types import SimpleNamespace

import numpy
from tqdm import tqdm

import cv2

from ultralytics import YOLO
# from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.engine.results import Results, Boxes

from classifier.Classifierv8 import Classifier
from plotter.Plotter import Plotter
from tracker.TrackInfo import TrackInfo, Rect

from PIL import Image

from tracker.tracker_sahi import SahiTurtleTracker, MockResults

def load_config_value(configuration: yaml, config_key: str, default_value: any) -> any:
    try:
        return configuration[config_key]
    except KeyError:
        return default_value


class Pipeline():
    def __init__(self, path_config_pipeline: str, path_config_tracker: Optional[str] = None, keep_clean_view: bool = False) -> None:
        with open(os.path.expanduser(path_config_pipeline), 'r') as file:
            configuration: yaml = yaml.safe_load(file)

        if configuration is None:
            raise Exception("Unable to load pipeline configuration!")
        
        if not path_config_tracker:
            path_config_tracker = "botsorttracker_config.yaml"
        
        self.path_config_tracker = os.path.expanduser(path_config_tracker)
        if not os.path.exists(self.path_config_tracker):
            raise Exception("Unable to load tracker configuration!")
        
        # TODO - This is hard coded as a BYTETracker, the parameters config should match and we should get the version Matt was using...
        # self.tracker = BYTETracker(args=self.path_config_tracker)
        self.tracker: BOTSORT = None

        try:
            self.all_detection_models: dict[str, str] = configuration['detection_models']
            self.all_classification_models: dict[str, str] = configuration['classification_models']
        except KeyError:
            raise Exception("Configuration incomplete: please specify both detection_models and classification_models!")
        
        if len(self.all_detection_models) == 0:
            raise Exception("No detection_models found in configuration!")
        
        if len(self.all_classification_models) == 0:
            raise Exception("No classification_models found in configuration!")
        
        self.assert_models_exist(self.all_detection_models)
        self.assert_models_exist(self.all_classification_models)

        self.video_name: str = ""
        self.video_start_offset_name_postfix: str = ""

        self.write_video: bool = load_config_value(configuration, "write_video", True)
        self.frame_skip: int = load_config_value(configuration, "frame_skip", 2)
        self.fps: float = 30.0
        self.keep_clean_view: bool = keep_clean_view
        self.detector_image_size: int = 640
        self.output_image_height: int = 720
        self.classifier_image_size: int = 64
        self.total_frames: int = 0

        self.frames_to_process_not_considering_skip: int = 0
        self.actual_frames_processed: int = 0
        self.frames_skipped: int = 0
        self.start_frame_index: int = 0
        self.processing_complete: bool = False

        self.video_in: Optional[cv2.VideoCapture] = None

        self.detection_model_name: str = ""
        self.classification_model_name: str = ""

        self.tracks: dict[int, TrackInfo] = {}
        self.tracks_updated: List[TrackInfo] = []
        self.plotter: Plotter = Plotter()

    def get_current_video_timestamp_seconds(self) -> float:
        return self.get_current_unprocessed_frame_index() / self.fps

    def assert_models_exist(self, model_dictionary: Dict[str, str]) -> None:
        for model in model_dictionary.items():
            expanded_path: str = os.path.expanduser(model[1])
            if not os.path.exists(expanded_path):
                raise Exception(f"Model '{model[0]}' does not exist at path '{expanded_path}'")

    def get_video_start_offset_name_postfix(self, start_processing_time_seconds: float) -> str:
        video_start_offset_name_postfix: str = ""

        if start_processing_time_seconds > 0:
            minutes: int = int(start_processing_time_seconds // 60)
            seconds: int = int(start_processing_time_seconds % 60)
            # Format in 00m_00s style.
            video_start_offset_name_postfix = f"_from_{minutes:02d}m{seconds:02d}s"

        return video_start_offset_name_postfix

    def setup(self, video_in_path: str, output_dir_path: str, detection_model_name: Optional[str] = None, classification_model_name: Optional[str] = None, start_processing_time_seconds: float = 0.0, end_processing_time_seconds: float = 0.0) -> None:
        self.tracks.clear()
        self.video_path = os.path.expanduser(video_in_path)
        self.output_dir_path = os.path.expanduser(output_dir_path)

        self.detection_model_name = detection_model_name

        if not detection_model_name:
            # Use the first model specified.
            detection_model_path = next(iter(self.all_detection_models.values()))
        else:
            detection_model_path = self.all_detection_models[detection_model_name]

        self.classification_model_name = classification_model_name

        if not classification_model_name:
            # Use the first model specified.
            classification_model_path = next(iter(self.all_classification_models.values()))
        else:
            classification_model_path = self.all_classification_models[classification_model_name]

        self.video_name: str = os.path.basename(self.video_path).rsplit('.', 1)[0]
        self.video_start_offset_name_postfix: str = self.get_video_start_offset_name_postfix(start_processing_time_seconds)
        csv_file_name = f"{self.video_name}_tracks{self.video_start_offset_name_postfix}.csv"
        self.output_tracks: str = os.path.join(self.output_dir_path, csv_file_name)

        os.makedirs(self.output_dir_path, exist_ok=True)

        self.model_track: YOLO = YOLO(os.path.expanduser(detection_model_path))
        self.model_track.fuse()
        try:
            self.TurtleClassifier: Classifier = Classifier(weights_file = classification_model_path,
                                           classifier_image_size=self.classifier_image_size)
        except Exception as e:
            print(f"Error loading classification model: {e}")
            raise e

        self.setup_video_read(start_processing_time_seconds, end_processing_time_seconds)
        
        if self.write_video:
            self.init_video_write()

        # TODO - Don't hard-code these values...
        self.sahi_target_h: int = 1440
        self.sahi_slice: int = 640
        self.sahi_overlap: float = 0.2
        self.sahi_conf: float = 0.3
        # tracker_args = SimpleNamespace(
        #     conf=self.sahi_conf, 
        #     track_thresh=numpy.clip(self.sahi_conf-0.2,a_min=0.1,a_max=0.9), 
        #     track_high_thresh=numpy.clip(self.sahi_conf+0.4, a_min=0.1, a_max=0.9),
        #     track_low_thresh=0.01, 
        #     new_track_thresh=numpy.clip(self.sahi_conf+0.5, a_min=0.1, a_max=0.9), 
        #     match_thresh=0.9,
        #     track_buffer=180, 
        #     frame_rate=self.fps, 
        #     mot20=False, 
        #     fuse_score=True,
        #     gating_thres=255, 
        #     proximity_thres=0.5, 
        #     appearance_thres=0.5
        # )
        # self.tracker = BYTETracker(args=tracker_args)

        tracker_args = SimpleNamespace(
            tracker_type= "botsort",
            track_high_thresh= 0.25,
            track_low_thresh= 0.1,
            new_track_thresh= 0.25,
            track_buffer= 30,
            match_thresh= 0.8,
            fuse_score= True,
            gmc_method= "sparseOptFlow",
            proximity_thresh= 0.5,
            appearance_thresh= 0.8,
            with_reid= False,
            model= "auto"
        )

        self.tracker = BOTSORT(args=tracker_args)


    def reset_to_beginning(self) -> None:
        self.actual_frames_processed = 0
        self.frames_skipped = 0
        self.processing_complete = False

        self.tracks.clear()
        self.tracks_updated.clear()

    def set_total_frame_count(self, total_frame_count: int, start_frame_index_from_time: int = 0, end_frame_index_from_time: int = 0) -> None:
        self.total_frames = total_frame_count
        print(f'Video frame count: {self.total_frames}')

        if end_frame_index_from_time > 0 and end_frame_index_from_time >= total_frame_count:
            end_frame_index_from_time = total_frame_count - 1

        if start_frame_index_from_time >= total_frame_count:
            raise Exception("Start processing time is too large, no frames left to process!")

        self.processing_complete = False

        self.start_frame_index = start_frame_index_from_time

        if end_frame_index_from_time <= 0:
            # No end offset.
            self.frames_to_process_not_considering_skip = self.total_frames - self.start_frame_index
        else:
            if end_frame_index_from_time <= self.start_frame_index:
                raise Exception("End processing time must be greater than start processing time!")
            
            self.frames_to_process_not_considering_skip = (end_frame_index_from_time - self.start_frame_index) + 1

        if self.frames_to_process_not_considering_skip <= 0:
            raise Exception("Start and end offsets are too large, no frames left to process!")

        self.actual_frames_processed = 0
        self.frames_skipped = 0

    def setup_video_read(self, start_processing_time_seconds: float = 0.0, end_processing_time_seconds: float = 0.0) -> None:        
        print(f'Video name: {self.video_name}')
        print(f'Video location: {self.video_path}')

        if self.video_in:
            self.video_in.release()

        self.video_in = cv2.VideoCapture(self.video_path)

        if not self.video_in.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
            
        # get fps of video
        self.fps = self.video_in.get(cv2.CAP_PROP_FPS)
        print(f'Video FPS: {self.fps}')

        start_frame_index_from_time: int = int(self.fps * start_processing_time_seconds)
        end_frame_index_from_time: int = int(self.fps * end_processing_time_seconds)
        
        # get total number of frames of video
        self.set_total_frame_count(int(self.video_in.get(cv2.CAP_PROP_FRAME_COUNT)), start_frame_index_from_time, end_frame_index_from_time)

        self.image_width = int(self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        print(f'Image width: {self.image_width}')
        print(f'Image height: {self.image_height}')

        image_ratio: float = float(self.image_width) / self.image_height
        self.dimensions_view: tuple[int, int] = (int(self.output_image_height * image_ratio), self.output_image_height)

        image_scale_factor: float = self.detector_image_size / self.image_width
        self.dimensions_processing: tuple[int, int] = (int(self.image_width * image_scale_factor), int(self.image_height * image_scale_factor))

        self.mat_original: numpy.ndarray = numpy.zeros([self.image_height, self.image_width, 3], dtype=numpy.uint8)
        self.mat_turtle_finding: numpy.ndarray = numpy.zeros([self.dimensions_processing[1], self.dimensions_processing[0], 3], dtype=numpy.uint8)
        self.mat_view_processed: numpy.ndarray = numpy.zeros([self.dimensions_view[1], self.dimensions_view[0], 3], dtype=numpy.uint8)
        self.mat_view_clean: Optional[numpy.ndarray] = numpy.zeros([self.dimensions_view[1], self.dimensions_view[0], 3], dtype=numpy.uint8) if self.keep_clean_view else None

    def find_tracks_in_frame(self, time: float, frame: numpy.ndarray, threshold_detection: float, threshold_tracking: float) -> None:
        '''Given an image as a numpy array, find and track all turtles.
        '''
        self.tracks_updated.clear()
        results: List[Results] = self.model_track.track(source=frame,
                                         stream=True, 
                                         persist=True, 
                                         show_boxes=True,
                                         verbose=False,
                                         conf=threshold_detection, # test for detection thresholds
                                         iou=threshold_tracking,
                                         tracker=self.path_config_tracker)
        
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes: Boxes = result.boxes

            for i, id in enumerate(boxes.id):
                track_id: int = int(id) # track_id starts at one :'(
                xyxyn: numpy.ndarray = numpy.array(boxes.xyxyn[i])
                latest_box: Rect = Rect(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
                confidence: float = float(boxes.conf[i])

                if track_id not in self.tracks.keys():
                    # Create a new track
                    new_track: TrackInfo = TrackInfo(track_id, time, latest_box, confidence)
                    self.tracks[track_id] = new_track
                    self.tracks_updated.append(new_track)
                else:
                    # Update existing track information
                    existing_track: TrackInfo = self.tracks[track_id]
                    existing_track.update_turtleness(latest_box, confidence)
                    self.tracks_updated.append(existing_track)

    def classify_turtles(self, frame: numpy.ndarray) -> None:
        (height, width, _) = frame.shape
        for track in self.tracks_updated:
            box: Rect = track.latest_box
            roi_left: int = int(box.left * width)
            roi_right: int = int(box.right * width)
            roi_top: int = int(box.top * height)
            roi_bottom: int = int(box.bottom * height)
            frame_cropped: numpy.ndarray = frame[roi_top:roi_bottom, roi_left:roi_right, :]
            cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB, frame_cropped)
            # The classifier does not work well if we aren't using a PIL Image
            pil_image: Image = Image.fromarray(frame_cropped)
            paintedness_confidence = self.TurtleClassifier.classify(pil_image)
            cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR, frame_cropped)
            track.update_paintedness(paintedness_confidence)

    def plot_data(self, frame: numpy.ndarray, threshold_classifier: float) -> None:
        # plotting onto the image with self.plotter
        for track in self.tracks_updated:
            self.plotter.draw_labeled_box(frame, track, threshold_classifier)

    def init_video_write(self) -> None:
        video_file_name = f"{self.video_name}_tracked{self.video_start_offset_name_postfix}.mp4"
        video_out_name = os.path.join(self.output_dir_path, video_file_name)
        self.video_out = cv2.VideoWriter(video_out_name, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   self.fps / self.frame_skip, 
                                   self.dimensions_view,
                                   isColor=True)
        
    def write_to_csv(self) -> None:
        header = ['track_id', 'turtle_confidences', 'marked_confidences', 'marked_confidence_mean']
        with open(self.output_tracks, mode='w', newline='') as csv_file:
            f = csv.writer(csv_file)
            f.writerow(header)
            for track in self.tracks.values():
                track_id = track.id
                turtleness = track.confidences_is_turtle
                paintedness = track.confidences_is_painted
                paintedness_avg = track.confidence_is_painted_mean

                f.writerow([track_id, turtleness, paintedness, paintedness_avg])

    def get_current_unprocessed_frame_index(self) -> int:
        return self.start_frame_index + self.actual_frames_processed + self.frames_skipped

    def process_frame(self, threshold_detection: float, threshold_tracking: float, threshold_classifier: float) -> bool:
        if self.processing_complete:
            return False

        index_to_process: int = self.get_current_unprocessed_frame_index()

        if (self.actual_frames_processed + self.frames_skipped) >= self.frames_to_process_not_considering_skip:
            print("All frames processed.")
            self.processing_complete = True
            return False

        if not self.video_in.isOpened():
            self.processing_complete = True
            raise Exception("Video has closed unexpectedly.")
        
        self.video_in.set(cv2.CAP_PROP_POS_FRAMES, index_to_process)
        read_result: Tuple[bool, numpy.ndarray] = self.video_in.read(self.mat_original)

        if not read_result[0]:
            print("Read result was false, likely end of video reached.")
            self.processing_complete = True
            raise Exception("Failed to read frame, likely end of video reached.")

        time: float = index_to_process / self.fps

        use_sahi: bool = True # TODO - Move this to the configuration
        if use_sahi:
            self.process_frame_sahi(time)
        else:

            cv2.resize(src=self.mat_original, dsize=self.dimensions_processing, dst=self.mat_turtle_finding)

            cv2.resize(src=self.mat_original, dsize=self.dimensions_view, dst=self.mat_view_processed)

            if self.keep_clean_view:
                numpy.copyto(src=self.mat_view_processed, dst=self.mat_view_clean)

            
            self.find_tracks_in_frame(time, self.mat_turtle_finding, threshold_detection, threshold_tracking)

        self.classify_turtles(self.mat_original)
        self.plot_data(self.mat_view_processed, threshold_classifier)
        
        if self.write_video:
            self.video_out.write(self.mat_view_processed)

        self.actual_frames_processed += 1
        self.frames_skipped += self.frame_skip - 1

        return True
    
    def process_frame_sahi(self, time: float) -> None:
        h_orig, w_orig = self.mat_original.shape[:2]
        ratio = self.sahi_target_h / h_orig

        # Resizing of images
        current_processing_dimensions = (int(w_orig * ratio), self.sahi_target_h)

        existing_turtle_finding_shape = self.mat_turtle_finding.shape[:2]
        if existing_turtle_finding_shape != current_processing_dimensions:
            self.dimensions_processing = current_processing_dimensions
            self.mat_turtle_finding: numpy.ndarray = numpy.zeros([self.dimensions_processing[1], self.dimensions_processing[0], 3], dtype=numpy.uint8)

        cv2.resize(src=self.mat_original, dsize=self.dimensions_processing, dst=self.mat_turtle_finding)

        cv2.resize(src=self.mat_original, dsize=self.dimensions_view, dst=self.mat_view_processed)

        if self.keep_clean_view:
            numpy.copyto(src=self.mat_view_processed, dst=self.mat_view_clean)

        # Perform the inference
        all_dets = SahiTurtleTracker.custom_sahi_inference(self.model_track, self.mat_turtle_finding, self.sahi_slice, self.sahi_overlap, self.sahi_conf)

        self.tracks_updated.clear()

        if len(all_dets) > 0:
            mock_results = MockResults(all_dets[:, :4], all_dets[:, 4], all_dets[:, 5])
            tracker_tracks: numpy.ndarray = self.tracker.update(mock_results, self.mat_turtle_finding)

            # Update the tracks.
            for t in tracker_tracks:
                x1, y1, x2, y2, t_id_input, confidence = t[:6]
                track_id = int(t_id_input)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                # self.tracks[tid].append((int(cx), int(cy)))

                # latest_box: Rect = Rect(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
                latest_box: Rect = Rect(
                    x1 / current_processing_dimensions[0], 
                    y1 / current_processing_dimensions[1], 
                    x2 / current_processing_dimensions[0], 
                    y2 / current_processing_dimensions[1]
                )
                # confidence: float = float(boxes.conf[i])
                # confidence: float = 0.5 # TODO - How do we get the confidence?

                if track_id not in self.tracks.keys():
                    # Create a new track
                    new_track: TrackInfo = TrackInfo(track_id, time, latest_box, confidence)
                    self.tracks[track_id] = new_track
                    self.tracks_updated.append(new_track)
                else:
                    # Update existing track information
                    existing_track: TrackInfo = self.tracks[track_id]
                    existing_track.update_turtleness(latest_box, confidence)
                    self.tracks_updated.append(existing_track)

                # TODO - Should we have this or not:
                # if len(self.tracks[tid]) > 30: self.tracks[tid].pop(0)

                # if len(self.tracks[tid]) > 5: count.add(tid)

        # self.classify_turtles(self.mat_original)
        # self.plot_data(self.mat_view_processed, threshold_classifier)

        # SahiTurtleTracker
    
# for i, id in enumerate(boxes.id):
#                 track_id: int = int(id) # track_id starts at one :'(
#                 xyxyn: numpy.ndarray = numpy.array(boxes.xyxyn[i])
#                 latest_box: Rect = Rect(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
#                 confidence: float = float(boxes.conf[i])

                


    def is_processing_complete(self) -> bool:
        return self.processing_complete
    
    def finish(self) -> None:
        cv2.destroyAllWindows()
        
        if self.write_video:
            self.video_out.release()

        self.video_in.release()

        self.write_to_csv()

    def run(self, threshold_detection: float, threshold_tracking: float, threshold_classifier: float, show_preview_window: bool) -> None:
        progress_bar: tqdm = tqdm(total=self.total_frames)

        while self.process_frame(threshold_detection, threshold_tracking, threshold_classifier):        
            if show_preview_window:
                cv2.imshow('images', self.mat_view_processed)
                if cv2.waitKey(1)& 0xFF == ord('q'):
                    print("Shutting down and saving data...")
                    break 
            
            progress_bar.update(self.frame_skip)
        
        self.finish()

    def detection_model_exists(self, detection_model_name: str) -> bool:
        return detection_model_name in self.all_detection_models.keys()
    
    def classifier_model_exists(self, classification_model_name: str) -> bool:
        return classification_model_name in self.all_classification_models.keys()


def get_kwargs(args: List[str]) -> Dict[str, str]:
    kwargs: Dict[str, str] = dict()

    for arg in args:
        split = arg.split(":=")
        if len(split) != 2:
            continue

        kwargs[split[0]] = split[1]
    
    return kwargs

def parse_bool(value: str) -> bool:
    return value.lower() in ["true", "yes", "y"]

def main() -> None:
    kwargs: Dict[str, str] = get_kwargs(sys.argv[1:])

    # Process required arguments
    try:
        video_in_path: str = kwargs["video_in_path"]
        output_path: str = kwargs["output_path"]
    except KeyError:
        print(f"Usage: {os.path.basename(__file__)} video_in_path:=/path/to/video/file output_path:=/path/to/output/directory/ [config:=/path/to/configuration.yaml]")
        return

    try:
        configuration_path: str = kwargs["config"]
    except KeyError:
        configuration_path: str = "sm_tracking_pipeline_config.yaml"

    try:
        show_preview_window: bool = parse_bool(kwargs["show_preview_window"])
    except KeyError:
        show_preview_window: bool = True

    pipeline: Pipeline = Pipeline(configuration_path)

    pipeline.setup(video_in_path, output_path)
    
    threshold_detection: float = 0.2
    threshold_tracking: float = 0.5
    confidence_threshold_classifier: float = 0.7

    pipeline.run(threshold_detection, threshold_tracking, confidence_threshold_classifier, show_preview_window)


if __name__ == "__main__":
    main()