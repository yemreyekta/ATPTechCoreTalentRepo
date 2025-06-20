
# 2025
ATPTech Core Talent Repo

# Camera Movement Detection Challenge

This project is a starter kit for detecting significant camera movement from a sequence of images using computer vision techniques.

## 📸 Challenge Overview

Build an algorithm that detects *significant movement* of a camera (e.g., shake, tilt, pan) by analyzing consecutive image frames.

**Your tasks:**
- Implement movement detection logic in `movement_detector.py`
- Create a simple web app interface in `app.py` for uploading images/videos and viewing results
- Deploy your solution (e.g., Streamlit Cloud or Hugging Face Spaces)
- Submit your app URL and GitHub repo

---

## 🚀 Getting Started

1. Clone this repo
2. Install dependencies:  
    pip install -r requirements.txt
3. Add or use sample frames in `test_images/`
4. Run locally:  

---

## 📝 Deliverables

- Publicly deployed app URL
- Updated GitHub repo (this one or your fork)
- Complete README with approach and instructions

---

## 📂 Files

- `movement_detector.py`: Put your main detection logic here
- `app.py`: Streamlit web app
- `requirements.txt`: Dependencies
- `test_images/`: Place sample image frames for testing

---

## 💡 Hints

- Check out OpenCV functions like `cv2.absdiff`, `cv2.goodFeaturesToTrack`, `cv2.findHomography`
- For extra credit: Visualize detected movement on output frames

---

Good luck!
