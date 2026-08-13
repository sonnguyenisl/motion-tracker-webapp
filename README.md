# AXIMOVE

> **AI-powered movement analysis and coaching.**

AXIMOVE is a web application that uses **computer vision** and **movement analysis** to help users evaluate their exercise technique. Users can upload a reference movement and their own movement, compare the two, and receive feedback based on their performance.

---

## Overview

AXIMOVE analyzes human movement from video rather than relying only on raw video comparison. The system extracts **body pose information**, analyzes movement patterns, and compares the user's performance with a reference movement.

The goal is to make movement analysis more accessible by providing users with **understandable and actionable feedback**.

---

## Features

- **Reference and user video upload**
- **Human pose estimation** using MediaPipe
- **Movement analysis** using body pose and joint angles
- **Movement comparison** using Dynamic Time Warping (DTW)
- **Repetition counting**
- **Range of Motion (ROM) analysis**
- **AI-powered coaching feedback**
- **Live camera movement analysis**
- **User accounts and analysis history**

---

## How It Works

### 1. Pose Estimation

AXIMOVE processes the input video and extracts **human body landmarks** using MediaPipe.

These landmarks can then be used to calculate movement-related features such as **joint angles** and **range of motion**.

### 2. Movement Analysis

The extracted pose data is converted into **movement sequences** that can be analyzed over time.

This allows AXIMOVE to focus on the user's actual movement rather than differences in backgrounds, lighting, or video appearance.

### 3. Movement Comparison

Users may perform the same exercise at different speeds. AXIMOVE uses **Dynamic Time Warping (DTW)** to align movement sequences and compare them despite differences in timing.

### 4. Coaching

The results of the movement analysis are used to generate **AI-powered feedback** that helps users understand potential issues with their exercise technique.

---

## Tech Stack

| **Category** | **Technology** |
|---|---|
| **Backend** | Python, Flask |
| **Computer Vision** | OpenCV |
| **Pose Estimation** | MediaPipe |
| **Movement Analysis** | tslearn, Dynamic Time Warping |
| **Frontend** | HTML, CSS, JavaScript |
| **Database** | SQLite |
| **Deployment** | Docker, Railway |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sonnguyenisl/motion-tracker-webapp.git
cd motion-tracker-webapp
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set the OpenRouter API Key

Set the `OPENROUTER_API_KEY` environment variable with your OpenRouter API key.

**Windows Command Prompt:**

```cmd
set OPENROUTER_API_KEY=your_api_key_here
```

**Windows PowerShell:**

```powershell
$env:OPENROUTER_API_KEY="your_api_key_here"
```

**macOS / Linux:**

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### 5. Run the Application

```bash
python run.py
```

The application should then be available at:

```text
http://127.0.0.1:1523
```

---

## Usage

1. **Create an account or log in.**
2. **Upload a reference exercise video.**
3. **Upload a video of your own movement.**
4. **Start the movement analysis.**
5. **Review the movement comparison and performance metrics.**
6. **View the generated coaching feedback.**

---

## Limitations

AXIMOVE's performance can be affected by factors such as:

* Camera angle and positioning
* Body occlusion
* Video quality and lighting
* Differences in body proportions
* Exercises that are not yet specifically supported

> **Note:** AXIMOVE is intended as an assistive movement-analysis tool and should not be considered a replacement for professional coaching or medical assessment.

---

## Future Development

Potential improvements include:

* Support for additional exercises
* Improved movement normalization
* More robust movement comparison
* More personalized performance analysis
* Improved real-time feedback
* Expanded exercise and movement library
* Mobile support

---

## Contributors

| **Name**         | **Role**  |
| ---------------- | --------- |
| **Nguyen Son**   | Developer |
| **Vu Tien Minh** | Developer |

---

## Repository

[**AXIMOVE GitHub Repository**](https://github.com/sonnguyenisl/motion-tracker-webapp)

[**AXIMOVE Official Website**](https://aximoveofficial.up.railway.app/)
