# SegmentationPipeline
A Python tool to process images using a background segmentation microservice. The tool resizes images, sends them to the microservice, and saves the output masks and resized images.

## Features
- Automatically resizes images based on their height.
- Sends images to a microservice for background segmentation.
- Saves processed images and masks in separate folders.
- Supports parallel processing for faster execution.

## Requirements
- Python 3.x
- OpenCV
- Requests
- NumPy
