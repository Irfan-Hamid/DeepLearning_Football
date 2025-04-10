# Deep Learning Pipeline for Football Match Analysis

## Overview
This project presents a complete deep learning pipeline for analyzing football match videos, inspired by the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) Kaggle competition. It automates the detection, tracking, and analysis of players and ball movements, performs team assignment, and provides tactical visualizations such as radar views and Voronoi diagrams to represent control and spatial dominance.

---

## Datasets
This project uses two datasets hosted on Roboflow for fine-tuning:

1. **[Football Player Detection Dataset](https://universe.roboflow.com/irfanworskspace/football_player_detection-zwwem/dataset/1)**  
   - Object Detection classes: `player`, `goalkeeper`, `referee`, `ball`

2. **[Football Field Keypoint Detection Dataset](https://universe.roboflow.com/irfanworskspace/football_field_keypoint_detection/dataset/1)**  
   - Keypoint Detection classes: 32 characteristic pitch points including `penalty area`, `goal area`, `center circle`, and `corners`

---

## Components

### A. `Train_Object_Detector_YOLO.ipynb`
This notebook fine-tunes a **YOLOv8** model for object detection on football matches.

- Input: The Roboflow 'Football Player Detection' dataset mentioned in the 'Datasets section'.
- Classes: `players`, `goalkeepers`, `referees`, `ball`
- Output: Custom-trained weights for use in inference

---

### B. `Train_KeyPoint_Detector_YOLO.ipynb`
This notebook fine-tunes a **YOLOv8** model for keypoint detection on the football pitch.

- Input: The Roboflow 'Football Field Keypoint Detection' dataset mentioned in the 'Datasets section'.
- Output: Trained model to predict 32 unique pitch landmarks

---

### C. `DeepLearning_Football.ipynb`  
This is the core analysis pipeline notebook. It processes football match videos and performs complete scene understanding.

---

#### 1. **Source Video Ingestion**
- Downloads and loads match videos from the **DFL Bundesliga Data Shootout** competition
- Splits videos into individual frames

---

#### 2. **Object & Keypoint Detection**
- Each frame is passed through the two fine tuned YOLOv8 models:
  - **Object Detector** → detects players, goalkeepers, referees, and the ball
  - **Keypoint Detector** → detects 32 pitch landmarks

---

#### 3. **Tracking (Player + Referee Only)**
- Applies **ByteTrack** to maintain unique identity across frames for each detected player and referee.

---

#### 4. **Team Assignment**
- Samples one frame per second
- Detect players within those frames and crop out detected players
- Embeddings are generated for the crops using **SigLIP**
- **UMAP** is used to reduce embeddings from 768D to 3D
- **KMeans** clustering groups players into two teams

---

#### 5. **Perspective Transformation**
Performs geometric transformation using keypoints:
- **Frame → Pitch** → for projecting player/ball positions onto a 2D pitch map
- **Pitch → Frame** → for overlaying virtual lines back on the actual match footage

---

##### (a). **Line Projection**
- Uses pitch keypoints to draw accurate virtual lines on video (e.g., halfway line, box boundaries)
- Accuracy depends on precision of the keypoint predictions

---

##### (b). **Player & Ball Projection**
- Projects tracked entities to a radar-style top-down pitch view
- Simulates tactical visuals similar to video game minimaps

---

#### 6. **Voronoi Diagram Generation**
- Calculates Voronoi zones for all players based on projected coordinates
- Shows team control zones on the pitch in real-time
- Useful for spatial and tactical analysis

---

##  Tools Used
- **YOLOv8** – Object and keypoint detection
- **SigLIP** – Vision language embeddings for team clustering
- **UMAP** – Dimensionality reduction
- **KMeans** – Unsupervised team separation
- **ByteTrack** – Object tracking
- **OpenCV** – Perspective transformation and drawing

---

## Output Features
- Accurate detection of all match participants
- Smooth multi-object tracking
- Team classification
- Radar style pitch projection for tactical analysis
- Real time Voronoi diagrams to assess pitch control

---

## To Run

- Clone the repository.
- Open and run the `DeepLearning_Football.ipynb` notebook in **Google Colab**.

> The other two notebooks (`Train_Object_Detector_YOLO.ipynb` and `Train_KeyPoint_Detector_YOLO.ipynb`) are **only required** if you want to fine-tune the YOLOv8 models yourself and host them on Roboflow.  
> You will need a [Roboflow account](https://roboflow.com/) and an **API key**, which you can securely add to your environment in Colab.

---
