Core Talent AI Coder Challenge: Camera Movement Detection Challenge

Title: Detecting Significant Camera Movement Using Image Recognition

Scenario: Imagine you are tasked with building a component for a smart camera system. Your goal is to detect significant movement—for example, if someone moves or tilts the camera or if the entire camera is knocked or shifted. This is different from simply detecting moving objects in the scene.

Requirements:
1. Input:
    * A sequence of images or frames (at least 10-20), simulating a fixed camera, with some frames representing significant camera movement (tilt, pan, large translation), and others showing a static scene or minor background/object motion.
    * (You may use public datasets, generate synthetic data, or simulate with your own webcam.)
        * e.g.: https://huggingface.co/datasets/syCen/CameraBench
2. Task:
    * Build an algorithm (Python preferred) that analyzes consecutive frames and detects when significant camera movement occurs.
    * Output a list of frames (by index/number) where significant movement is detected.
3. Expected Features:
    * Basic: Frame differencing or feature matching to detect large global shifts (e.g., using OpenCV’s ORB/SIFT/SURF, optical flow, or homography).
    * Bonus: Distinguish between camera movement and object movement within the scene (e.g., use keypoint matching, estimate transformation matrices, etc.).
4. Deployment:
    * Wrap your solution in a small web app (Streamlit, Gradio, or Flask) that allows the user to upload a sequence of images (or a video), runs the detection, and displays the result.
    * Deploy the app on a public platform (Vercel, Streamlit Cloud, Hugging Face Spaces, etc.)
5. Deliverables:
    * Public app URL
    * GitHub repo (with code and requirements.txt)
    * README (explaining your approach, dataset, and how to use the app)
    * AI Prompts or Chat History

Evaluation Rubric
Criteria	Points	Details
Correctness	5	Accurately detects significant camera movement; low false positives/negatives.
Implementation	5	Clean code, good use of OpenCV or relevant libraries, modular structure.
Deployment	5	App is online, easy to use, and functions as described.
Innovation	3	Advanced techniques (feature matching, transformation estimation, clear object vs camera).
Documentation	2	Clear README, instructions, and concise explanation of method/logic.
Bonus: Include sample test images, demo video, or extra features (e.g., sensitivity slider, result visualization).

Suggested Stack
* Python or C#
* OpenCV for computer vision
* Streamlit, Gradio or shadcn powered Vercel site for quick web UI
* GitHub for code repo, Streamlit Cloud, Hugging Face Spaces, Vercel for deployment

Sample Instructions to Candidates
Instructions:
1.      Fork this repository or start your own.
2.      Build your movement detection algorithm in movement_detector.py.
3.      Develop a simple web app for uploading images/sequences and viewing results.
4.      Deploy the app and share your URL and GitHub repo link.
5.      Include a README.md that explains your approach, setup, and sample results.
Deadline: 27.06.2025 
Plagiarism: AI powered individual work only; use open-source libraries as needed but cite appropriately. 

Sample README Outline
* Overview of your approach and movement detection logic
* Any challenges or assumptions
* How to run the app locally
* Link to the live app
* Example input/output screenshots
