#!/usr/bin/env python3
"""Step through a recorded driving dataset and inspect it sample by sample.

Prints the label for each frame and shows the image with the commanded velocity
drawn on top. Press ``q`` / ``ESC`` to quit, any other key for the next sample.
"""

import os
import pickle

import cv2


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, "training_data.pkl")

    print(f"Loading dataset: {filename}")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    images = data["images"]
    labels = data["labels"]

    print("======================================")
    print(" Training Data Viewer")
    print("======================================")
    print(f"Number of images : {len(images)}")
    print(f"Number of labels : {len(labels)}")
    print("======================================")

    if len(images) != len(labels):
        print("WARNING: Image/label count mismatch!")

    for i, image in enumerate(images):
        linear_x, angular_z = labels[i][0], labels[i][1]

        print(f"Sample {i:05d} | linear.x = {linear_x:.4f} | angular.z = {angular_z:.4f}")

        display_image = image.copy()
        cv2.putText(display_image, f"Sample: {i}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_image, f"linear.x: {linear_x:.3f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_image, f"angular.z: {angular_z:.3f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Training Data", display_image)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            print("Viewer stopped.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
