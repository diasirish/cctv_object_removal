# README for Person Replacement on CCTV Cameras

## Project Description

This project is designed to replace a person detected on CCTV cameras with any other object or image specified by the user. The initial need was to obscure a few individuals appearing on the cameras, and this script was created to achieve that.

## Technologies Used

- **OpenCV**: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products. The library has more than 2500 optimized algorithms, including a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.

- **Detectron2**: Detectron2 is Facebook AI Research's next-generation software system that implements state-of-the-art object detection algorithms. It is a complete rewrite of the first Detectron and adds several new capabilities. It supports a variety of models including Mask R-CNN, Fast R-CNN, and RetinaNet. Detectron2 is designed to be flexible, extensible, and to provide fast training and inference times.

Initially, YOLO v5 (You Only Look Once version 5) was considered for this project due to prior experience with it, but ultimately Detectron2 was chosen for its superior capabilities in handling the specific requirements of this project.

## Example Images

Before processing (camera view of a person):
![Before Image](path/to/before_image.jpg)

After processing (person replaced with a specified object):
![After Image](path/to/after_image.jpg)

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```
   cd <project_directory>
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your input video or images and place them in the `input` directory.
2. Place the object or image you want to use for replacement in the `replacements` directory.
3. Run the script with the following command:
   ```
   python main.py --input input/video.mp4 --replacement replacements/object.png
   ```

## Configuration

- `--input`: Path to the input video or image.
- `--replacement`: Path to the replacement object or image.
- Additional configuration parameters can be set in the `config.json` file.

## TO DO

1. **Try usign YOLO v8 model**: Explore and integrate a Yolo v8 model in this project.
2. **Improve replacement accuracy**: Enhance the accuracy of the replacement process to make it more seamless and natural-looking.
3. **Optimize algorithms to increase speed**: Find ways to optimize this solution to speed up the processes.
4. **Create a parameter to choose ammount of windows to display for the user**: Should include options of 1 - frames with replacement, 2 - original and replacement,  4 - add object detection window and original background subtractions.  
5. **Add cuda support**: for the users who have the GPUs available to them.
6. **Documentation and Tutorials**: Expand documentation and provide detailed tutorials for setting up and using the project effectively.
7. **More ideas to come**: ...

For any questions or contributions, feel free to open an issue or pull request on GitHub.

---
