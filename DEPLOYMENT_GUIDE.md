# Multi-Laptop Deployment Guide 🚀

To run the **Sign Language Kiosk** across two different laptops (one for the User/Kiosk and one for the Employee/Staff), follow these steps:

### 1. Network Requirements
- **Same Wi-Fi**: Both laptops must be connected to the **same local network** (Wi-Fi or Ethernet).
- **Firewall**: Ensure that port **8000** (backend) and **5173** (frontend) are allowed through the firewall on the "Host" laptop.

---

### 2. The "Host" Laptop (Backend & Server)
Choose one laptop to be the "Host". This laptop will run the Python backend and the React development server.

1.  **Find your Local IP Address**:
    - Open Command Prompt (cmd) and type `ipconfig`.
    - Look for "IPv4 Address" (e.g., `192.168.1.15`).
2.  **Start the Backend**:
    - `cd mvp/backend`
    - `python main.py`
3.  **Start the Frontend**:
    - `cd frontend`
    - `npm run dev -- --host` (The `--host` flag allows other devices to connect).

---

### 3. The "Client" Laptop (Employee/Staff)
This laptop only needs a web browser.

1.  **Open the Browser**.
2.  **Navigate to the Host's IP**:
    - Type `http://<HOST_IP>:5173/login` (Replace `<HOST_IP>` with the IP found in Step 2, e.g., `http://192.168.1.15:5173/login`).
3.  **Select "Employee" Role** and log in.

---

### 4. How it Works (Technical)
- The **React Frontend** on both laptops will connect to the **Socket.io** server running at `http://<HOST_IP>:8000`.
- The **Session ID** is synchronized across the network. When the Kiosk laptop detects a user, the Employee laptop (anywhere on the Wi-Fi) will instantly receive the Handshake Request.
- **Voice/Text Messages** and **AI Recognition** events are broadcasted in real-time via the host server.

---

### 5. Troubleshooting
- **Connection Refused**: Check if `main.py` is running on `0.0.0.0` (it is by default in this project).
- **No Video Feed**: Ensure the Kiosk laptop has granted camera permissions in the browser.
- **Latency**: If the video is laggy, ensure both laptops have a strong Wi-Fi signal. Using a wired Ethernet connection is recommended for the Host.
