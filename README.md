# 🚗 Road Accident Detection System

This project is a road accident detection system using computer vision and deep learning models. It detects and tracks vehicles and accidents from video input.

---

## ⚙️ Setup Instructions
It is recommended to download the ZIP file from this repository and open it in a code editor like Visual Studio Code. Navigate to the root folder that contains all the system files and start running the commands in your terminal.

---

## 💻 Terminal Commands: 
## 1. **Create a Virtual Environment**
#### Windows
``` python -m venv venv```

## 2. **Activate the environment**
#### Windows
```.\venv\Scripts\activate``` <br>

Once activated, your terminal will look like: <br>
```(venv) PS C:\Users\Wong Jia Mien\Downloads\CP2_RoadAccidentDetection-main\CP2_RoadAccidentDetection-main>```

## 3. Install All requirements
```pip install --upgrade pip``` <br>
```pip install -r requirements.txt```

## 4. **Clone the YOLOv5 Repository and Install Dependencies**
```git clone https://github.com/ultralytics/yolov5.git ``` <br>
```cd yolov5``` <br>
```pip install -r requirements.txt``` <br>

## 5. **Return to Root Directory and Run the Program**
```cd ..``` <br>
``` python main.py```

## ✏️ Drawing Violation Area
When the system starts, it will prompt you to draw a restricted area on the screen. This area is used to detect vehicle violations. If a vehicle enters this area, a violation message will be displayed. <br>
#### To draw the area:
1. Click 4 points in clockwise order starting from the top-left corner of your desired polygon. <br>
2. After drawing, close the window to proceed.<br>
3. A confirmation window will show the drawn area.<br>
4. Close it again to start video analysis.<br>
<img width="1107" height="836" alt="image" src="https://github.com/user-attachments/assets/30f75a59-faf5-40e6-af90-3df995df2afa" /> 
<img width="1072" height="665" alt="image" src="https://github.com/user-attachments/assets/e6479ec3-e90f-4978-bc23-d9c26903ff8b" />

 ## 📊 Sample Video Analysis
<img width="1262" height="776" alt="image" src="https://github.com/user-attachments/assets/6eef4f8c-2f66-4a04-b630-24239aebf1e0" />

## 🎞️ Changing Test Cases

There are a total of 10 sample videos available for testing, organized into the following structure:
```text
RoadAccidentDetection/
├── TC1
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mp4
├── TC1_output
│   ├── video1_output.mp4
│   ├── video2_output.mp4
│   ├── video3_output.mp4
├── TC2
│   ├── video1.mp4
│   ├── video2.mp4
├── TC2_output
│   ├── video1_output.mp4
│   ├── video2_output.mp4
├── TC3
│   ├── video1.mp4
│   ├── video2.mp4
├── TC3_output
│   ├── video1_output.mp4
│   ├── video2_output.mp4
├── TC4
│   ├── video1.mp4
│   ├── video2.mp4
├── TC4_output
│   ├── video1_output.mp4
│   ├── video2_output.mp4
├── TC5
│   ├── video1.mp4
│   ├── video2.mp4
├── TC5_output
│   ├── video1_output.mp4
│   ├── video2_output.mp4
```
---

## ⚠️  Warning: The test videos contain scenes of car crashes that may be distressing to some viewers.

---

## 📄 Notes:
For demonstration purposes, all test videos have already been analyzed once. Their outputs are saved in the corresponding output folders. <br>
To analyze a different video, simply modify the input and output paths in the script. <br>
<img width="2253" height="818" alt="image" src="https://github.com/user-attachments/assets/b22b72fc-f176-4cab-b2b3-18c8a711b007" />







