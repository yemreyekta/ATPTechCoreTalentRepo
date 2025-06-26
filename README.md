# ATP Core Talent 2025
# Core Talent AI Coder Challenge: Camera Movement Detection

**Detecting Significant Camera Movement Using Image Recognition**

---

## Overview

This project implements a robust solution for detecting **significant camera movement** (such as tilting, panning, or shifting the entire camera) as opposed to just object movement within a scene. The solution uses computer vision techniques to distinguish between global (camera) and local (object) motion, and provides a user-friendly web interface for testing and demonstration.

---

## Approach & Movement Detection Logic

- **Frame Differencing & Optical Flow:**
  - The algorithm analyzes consecutive frames using frame differencing and dense optical flow (Farneback) to estimate global motion.
  - If the average optical flow magnitude exceeds a threshold, the frame is marked as having significant camera movement.
- **Object Movement Detection:**
  - Background subtraction (MOG2) and contour analysis are used to detect moving objects within the scene.
  - If large enough contours are found, the frame is marked as containing object movement.
- **Visualization:**
  - Detected camera movements are visualized with flow vectors.
  - Object movements are visualized with bounding boxes and contours.
- **Efficiency:**
  - Annotated frames are displayed immediately and not stored in memory, preventing RAM overload for long videos.
  - Detected indices are updated live at the top of the app.

---

## Challenges & Assumptions

- **Assumptions:**
  - Input is a sequence of images or a video simulating a fixed camera, with some frames containing significant camera movement.
  - Minor object/background motion should not trigger camera movement detection.
- **Challenges:**
  - Distinguishing between global (camera) and local (object) motion, especially in scenes with both.
  - Efficient memory usage for long videos or large image sequences.
  - Providing real-time feedback in the web interface.

---

## How to Run the App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yemreyekta/ATPTechCoreTalentRepo
   cd ATPTechCoreTalentRepo/camera-movement-detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
4. **Usage:**
   - Upload a video (mp4/gif) or a sequence of images.
   - The app will analyze the frames and display detected camera/object movements and their indices live.

---

## Live App

[Live Demo (Streamlit Cloud)] https://atptechcoretalentrepo-yunusemreyekta.streamlit.app/

---

## Example Input/Output Screenshots

- **Input:**
  - Video or image sequence simulating camera shake, pan, or tilt.
- **Output:**
  - List of frame indices with detected camera/object movement (displayed live at the top).
  - Visualizations of detected movements (optical flow for camera, bounding boxes for objects).

*Add screenshots here*

---

## Example Results

| Input | Camera Movement Visual | Object Movement Visual |
|-------|-----------------------|-----------------------|
| ![Input Example](camera-movement-detection\sample_video\shaking_timed_panning_output.mp4) | ![Camera Movement](camera-movement-detection\images\fc72c2c62ca44a1642a7f6281b96caa1500fa98cd8ebfd57f2bd7b6c.jpg) | ![Object Movement](camera-movement-detection\images\e9eb0d02ec2e4d6da53232ded94b25445cfc4d3eeb21b45d866b5272.jpg) |

---

## AI Prompts / Chat History

- This project was developed with the support of AI prompts.
1. Refactor the function detect_significant_movement to follow cleaner functional separation. Add preprocessing (e.g., grayscale conversion), frame differencing, and scoring as separate functions. Include docstrings and type hints for all new functions.
2. Implement a function using OpenCV ORB to compute keypoints and descriptors for each frame. Then compute matches between consecutive frames, estimate homography, and use the transformation magnitude to decide significant camera movement.
Return the index of frames where global motion exceeds a configurable threshold.
3. Enhance the movement detection by distinguishing object movement from camera motion.
Use keypoint matching and homography estimation between frames. If homography indicates global movement and frame differencing suggests local changes, classify the movement as camera-induced.
Return two separate lists: one for significant camera movement, another for local object changes.
4. Overlay matched keypoints or transformation vectors on the frames where significant camera movement is detected.
Create a helper function that annotates the image with visual feedback (e.g., matched points or bounding boxes), and return it alongside the detection result.
5. Enhance the Streamlit UI to include:
- A sidebar to adjust the movement threshold interactively
- A toggle to display visualized frames with overlays (e.g., keypoints or differences)
- A download button for exporting the list of detected frames
Make sure the app handles variable image sizes and grayscale inputs.
6. Write a README file according to developments.

---

## Citation & Credits

- Uses OpenCV, Streamlit, and PIL.
- Inspired by the [CameraBench Dataset](https://huggingface.co/datasets/syCen/CameraBench) for testing.
- https://docs.opencv.org/3.4/d5/dab/tutorial_sfm_trajectory_estimation.html
- https://hackmd.io/@lKuOpplzSUWLhLim2Z7ZJw/ryTpNXeGn
