"""Object detection module."""
import os
import time

import numpy as np
import tensorflow as tf


class ObjectDetector(object):
    """Wrapper of a tensorflow object detection model."""

    def __init__(self, root, device):
        """Load a tensorflow model from root."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_visible_devices(gpus[device], 'GPU')
        # tf.config.experimental.set_memory_growth(gpus[device], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[device], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        model_dir = os.path.join(root, "saved_model")
        model = tf.saved_model.load(str(model_dir))
        self.model = model.signatures['serving_default']

    def infer(self, image):
        # TODO: support batch inference
        """Run object detection on a single image."""
        image = np.asarray(image)
        resolution = image.shape
        # The input needs to be a tensor, convert it using
        # `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with
        # `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        start = time.time()
        output_dict = self.model(input_tensor)
        t_used = time.time() - start

        # All outputs are batches tensors.
        # Convert to numpy arrays, and
        # take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        updated_boxes = []
        for box in output_dict['detection_boxes']:
            updated_boxes.append(
                [box[1]*resolution[1], box[0]*resolution[0],
                 box[3]*resolution[1], box[2]*resolution[0]])
        output_dict['detection_boxes'] = np.array(updated_boxes)

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict[
            'detection_classes'].astype(np.int64)

        return output_dict, t_used
