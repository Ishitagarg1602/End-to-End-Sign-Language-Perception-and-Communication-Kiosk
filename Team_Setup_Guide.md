# Team Setup Guide: End-to-End Sign Language Kiosk

This guide covers everything you need to clone the repository, set up the environment, run the training pipeline on your own laptop, and launch the kiosk interfaces.

---

## 1. Prerequisites
Ensure you have the following installed on your laptop:
- **Git** (for version control)
- **Python 3.10** or higher
- **Node.js** (for running the React frontend)
- **OpenAI API Key** (for NLP Intent generation. Ask the team lead if you don't have this).

---

## 2. Clone the Repository
Open your command prompt or terminal and run:
```bash
git clone https://github.com/Ishitagarg1602/End-to-End-Sign-Language-Perception-and-Communication-Kiosk.git
cd End-to-End-Sign-Language-Perception-and-Communication-Kiosk
```

---

## 3. Download the Dataset
Since the raw videos are too massive for GitHub, they must be downloaded manually.
1. Download the Dataset `.zip` file from the team's Google Drive: `[INSERT GOOGLE DRIVE LINK HERE]`
2. Extract the zip file and place the contents exactly inside the `mvp/dataset/` folder.
   - It should look like this: `mvp/dataset/hello/...`, `mvp/dataset/wait/...`, etc.

---

## 4. Train the AI Model Locally
Now that you have the raw videos, you need to extract the hand/body landmarks, augment them for variety, and train the neural network.

Open a terminal in the root folder of the project (`End-to-End-Sign-Language-Perception-and-Communication-Kiosk`) and run:

**A. Set up the Python Environment:**
```bash
python -m venv isl_env
isl_env\Scripts\activate
pip install -r mvp/backend/requirements.txt
pip install -r data_pipeline/requirements_data.txt
```

**B. Run the Data Pipeline:**
This step converts the raw MP4 videos into numerical coordinates mapping the hands.
```bash
python data_pipeline/1_extract_landmarks.py
```
*(This may take 15-30 minutes depending on your CPU/GPU).*

**C. Augment the Data:**
This generates variations (speed, angles) to make the AI more robust.
```bash
python data_pipeline/2_augment.py
```

**D. Train the Neural Network:**
This analyzes the numerical data and creates the final `.pt` weights file.
```bash
cd mvp
python train_model_v2.py
cd ..
```
*When finished, you should see a `model_cnn_bilstm.pt` file generated inside the `mvp` folder.*

---

## 5. Run the Backend Server
The backend handles real-time camera streaming, the trained AI model mapping, and the OpenAI logic.

1. Navigate to the backend directory:
   ```bash
   cd mvp/backend
   ```
2. Create a file named `.env` in this directory and add the OpenAI key:
   ```text
   OPENAI_API_KEY=sk-your-openai-key-here
   ```
3. Start the Python server:
   ```bash
   python main.py
   ```
*(Leave this terminal running!)*

---

## 6. Run the Frontend Dashboard
The frontend contains the Kiosk view with the 3D robot avatar, and the Employee view.

1. Open a **new** terminal window and go to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install the User Interface libraries:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm run dev
   ```

---

## 7. Test the Application
Open your web browser and go to:
- **Kiosk Interface:** `http://localhost:5173/kiosk` (The user signs here)
- **Employee Interface:** `http://localhost:5173/employee` (Review & approve here)

**Test Flow:**
1. Show your hands to the Kiosk camera. It will prompt the Employee dashboard with a connection request.
2. Go to the Employee dashboard and accept the request.
3. Sign a word to the Kiosk, click "Done Signing", select the AI-generated intent, and hit Send!
