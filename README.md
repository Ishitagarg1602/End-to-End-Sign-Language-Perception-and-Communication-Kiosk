---
title: Sign Language Kiosk Backend
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
<div align="center">
  
# 🤟 End-to-End Sign Language Perception & Communication Kiosk

*A real-time, bidirectional communication system breaking the language barrier between Deaf individuals and bank employees.*

[![Frontend Status](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](#)
[![Backend Status](https://img.shields.io/badge/Backend-Hugging_Face-yellow?logo=huggingface)](#)
[![Model Accuracy](https://img.shields.io/badge/Model_Accuracy-99.91%25-brightgreen)](#)
[![Inference Speed](https://img.shields.io/badge/Latency-under_100ms-blue)](#)
[![Parameters](https://img.shields.io/badge/Model_Size-3.01_MB-orange)](#)

</div>

---

Imagine walking into a bank and not being able to ask a simple question because of a language barrier. For millions of Deaf individuals, this is an everyday reality. 

We built this **Sign Language Kiosk** to change that. 

This project isn't just a proof-of-concept—it's a fully functional, production-deployed system that allows Deaf users to communicate seamlessly using Indian Sign Language (ISL), without needing a human interpreter by their side. 

---

## 🌟 Why this project stands out

We obsessed over speed, accuracy, and user experience. Here's what makes this system powerful:

* **Insane Accuracy (99.91%):** Our custom **CNN-BiLSTM + Attention** model achieves near-perfect recognition across **78 complex, banking-specific ISL signs**.
* **Lightning Fast:** Nobody wants to wait for a translation. By combining Google MediaPipe hand landmarking with our extremely lightweight 3.01 MB PyTorch model, we brought end-to-end inference latency to **under 100 milliseconds**. 
* **True Bidirectional Conversation:** 
  * *Deaf User ➡️ Employee:* The kiosk captures 30-frame ISL gestures, translates them, and vocalizes the text for the employee.
  * *Employee ➡️ Deaf User:* We leverage **OpenAI Whisper** for state-of-the-art Speech-to-Text, converting the employee's spoken response instantly into readable text and visual cues.
* **Continuous Learning (Feedback Loop):** AI models drift, so we built ours to learn. The employee dashboard features a "Correct/Wrong" feedback mechanism. If a gesture is misclassified, the data is automatically saved, and the model is iteratively retrained to get smarter every single day.
* **Polished User Experience:** The kiosk features beautiful, custom GIF-based interaction screens that guide users naturally, ensuring an intuitive and welcoming experience.

---

## 🚀 Live Deployments

We split the architecture to ensure maximum scalability and zero bottlenecks. You can find our live deployments below:

### 💻 Frontend: Deployed on [Vercel](https://vercel.com)
* **Tech Stack:** React, Vite, Socket.IO
* **What it does:** Powers the stunning kiosk UI, manages the camera stream (capturing exactly 30 frames per gesture), renders custom dynamic visuals, and maintains a persistent, low-latency WebSocket connection.

### ⚙️ Backend: Deployed on [Hugging Face](https://huggingface.co/) Spaces (Docker)
* **Tech Stack:** FastAPI, PyTorch, MediaPipe, Python, OpenAI Whisper
* **What it does:** The heavy lifter. It receives coordinate streams via Socket.IO, extracts 126 features per frame (21 keypoints × 2 hands × 3 coordinates), runs our 784k parameter inference model, and fires the predicted banking term back to the frontend instantly.

---

## 🧠 The AI Under the Hood

If you're an ML enthusiast, here’s a peek into our journey:

We initially tested a baseline K-Nearest Neighbors (KNN) model, but it struggled at a mere 10.20% accuracy because it couldn't capture the complex temporal flow of sign language. 

So, we engineered a custom **CNN-BiLSTM + Attention Architecture (v2)**:
- **Topology:** `Conv1D ➡️ BatchNorm ➡️ BiLSTM (2 layers) ➡️ Attention Mechanism ➡️ Dense ➡️ Softmax`
- **Why Attention?:** The attention layer dynamically learns to focus on the most discriminative frames within a sequence, which is absolutely critical for distinguishing similar signs (like "Deposit" vs. "Withdrawal").
- **Custom-Built Dataset:** We didn't rely on pre-existing generic datasets. We painstakingly built our own high-quality dataset from scratch, carefully recording and preprocessing over 4,200+ gesture sequences to ensure maximum real-world reliability specifically for banking scenarios.
- **Training Strategy:** We trained on this custom data using heavy online augmentation (noise injection, temporal stretching, rotation, mirror, scaling, frame dropout) and Mixup to aggressively prevent overfitting.
- **The Result:** We boosted accuracy by a massive **879%** over the baseline, achieving **100% Validation Accuracy** and **99.91% Test Accuracy**.

---

## 🛠️ How to Run Locally

Want to spin this up on your own machine? It’s super easy.

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Ishitagarg1602/End-to-End-Sign-Language-Perception-and-Communication-Kiosk.git
   cd End-to-End-Sign-Language-Perception-and-Communication-Kiosk
   ```

2. **One-Click Startup (Windows):**
   Simply double-click the `start_kiosk.bat` file in the root directory! It will automatically handle starting up both the backend server and the frontend development environment.

3. **Manual Startup:**
   * **Backend:** Navigate to `mvp/backend`, install dependencies from `requirements.txt`, and run the FastAPI server.
   * **Frontend:** Navigate to `frontend`, run `npm install`, and start the Vite dev server with `npm run dev`.

---

## 🤝 The Future of Inclusive Banking

We built this project to prove that high-performance AI can directly improve accessibility in our daily lives. 

Feel free to explore the code, test the live deployments, and contribute! Whether it's adding new ISL signs or optimizing the UI, every PR brings us one step closer to a world where technology speaks everyone's language. 🌍

*Built with ❤️ for accessibility and inclusion.*
