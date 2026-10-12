"""Shared image constants and preprocessing/sequence helpers.

Both the recorder and the inference nodes must see the camera in exactly the same
geometry, so the input size lives here as a single source of truth:

    preprocess   : raw camera frame -> 75x75 RGB->YUV tensor
    augment      : training-time photometric/geometric jitter, with the label
                   adjusted whenever the steering angle is affected
    create_sequences : flat samples -> (N, T, H, W, C) windows for the RNN
"""

import pickle

import cv2
import numpy as np

SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 8, 75, 75, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Fraction of training frames (taken at random) that receive augmentation, and
# the translation-to-steering coupling used by the Udacity driving simulator.
AUGMENT_PROBABILITY = 0.6
TRANSLATE_STEERING_GAIN = 0.002


def resize(image):
    """Resize a camera frame to the model's input resolution."""
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def rgb2yuv(image):
    """Convert RGB to YUV, the colour space used by the behavioural-cloning CNN."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """Resize and recolour a frame; the inverse of nothing — applied identically
    at training and inference time."""
    return rgb2yuv(resize(image))


def random_flip(image, command):
    """Mirror the frame horizontally and invert the steering angle to match."""
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        command = command.copy()
        command[1] = -command[1]
    return image, command


def random_translate(image, command, range_x=100, range_y=10):
    """Shift the frame by a random offset and correct the steering angle.

    A horizontal shift changes where the road appears relative to the camera, so
    the corresponding steering correction is folded back into the label.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)

    command = command + trans_x * TRANSLATE_STEERING_GAIN

    trans_m = np.float32([[1, 0, trans_x],
                          [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, command


def random_shadow(image):
    """Darken a random triangular wedge of the frame to emulate roadside shade."""
    h, w = image.shape[:2]

    x1, y1 = w * np.random.rand(), 0
    x2, y2 = w * np.random.rand(), h

    xm, ym = np.mgrid[0:h, 0:w]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # Randomly keep either the wedge or its complement, and scale saturation
    # rather than brightness so the shadow stays physically plausible.
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """Scale brightness by up to ±20% to cover changing light conditions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(image, command, range_x=100, range_y=10):
    """Apply the full augmentation chain; returns the frame and its label."""
    image, command = random_flip(image, command)
    image, command = random_translate(image, command, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, command


def prepare_data(images, labels, is_training=False):
    """Preprocess a batch of frames, augmenting a random subset when training.

    Augmentation is applied to the *raw* frame, before resizing, so warping and
    shading happen at full camera resolution.
    """
    processed_images = np.empty(
        (len(images), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
    processed_labels = np.asarray(labels, dtype=np.float32).copy()

    for i in range(len(images)):
        image = np.asarray(images[i])
        command = processed_labels[i].copy()

        if is_training and np.random.rand() < AUGMENT_PROBABILITY:
            image, command = augment(image, command)

        processed_images[i] = preprocess(image)
        processed_labels[i] = command

    return processed_images, processed_labels


def load_pickle(filename):
    """Load an ``{"images": [...], "labels": [...]}`` recording and report its size."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    images, labels = data["images"], data["labels"]

    print(f"{filename}")
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")

    assert len(images) == len(labels)

    return images, labels


def create_sequences(images, labels, sequence_length):
    """Sliding-window the samples into ``(N, T, H, W, C)`` inputs.

    Each window predicts the label of its *last* frame, which is what the robot
    needs at inference time: a command for the current state of the world.
    """
    num_sequences = len(images) - sequence_length + 1

    X = np.empty((num_sequences, sequence_length,
                  IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
    y = np.empty((num_sequences, 2), dtype=np.float32)

    for i in range(num_sequences):
        X[i] = images[i:i + sequence_length]
        y[i] = labels[i + sequence_length - 1]

    return X, y
