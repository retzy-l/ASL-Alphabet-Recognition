# 🖐 ASL Alphabet Recognition

> Real-time American Sign Language recognition using a custom CNN and computer vision.

**Course:** CENG 476 — Deep Learning
**Team:** Abdirahman Ahmed Hussein · İlter Karamüftüoğlu



![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)




![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)




![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)



---

## Overview

An assistive communication tool that recognizes ASL hand signs in real time via webcam. The system classifies 29 letter classes, auto-types recognized letters, and speaks sentences aloud — designed to bridge communication for the deaf and hard-of-hearing community.

---

## Model Architecture

Custom CNN trained on 64×64 RGB images across 29 classes.

```text
Input (64×64 RGB)
  → Conv(3→32)   → BatchNorm → ReLU → MaxPool
  → Conv(32→64)  → BatchNorm → ReLU → MaxPool
  → Conv(64→128) → BatchNorm → ReLU → MaxPool
  → Flatten → Dropout(0.5)
  → Linear(8192→512) → ReLU → Dropout(0.5)
  → Linear(512→29)
