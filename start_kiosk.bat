@echo off
echo ========================================================
echo ISL Banking Kiosk - Startup Script
echo ========================================================
echo.

:: Check if frontend needs to be built
if not exist "frontend\dist\index.html" (
    echo [1/3] Building React Frontend...
    cd frontend
    call npm run build
    cd ..
    echo Frontend built successfully.
) else (
    echo [1/3] React Frontend already built. Skipping build.
)

echo.
echo [2/3] Starting FastAPI Backend on port 8001...
start "ISL Backend Server" cmd /c "python -m uvicorn mvp.backend.main:app --host 0.0.0.0 --port 8001"

:: Wait a few seconds for the backend to initialize
timeout /t 5 /nobreak > nul

echo.
echo [3/3] Starting Public Tunnel (LocalTunnel)...
echo.
echo ========================================================
echo ALL SET! 
echo.
echo Employee Dashboard: https://isl-banking.loca.lt/employee
echo Kiosk UI:           https://isl-banking.loca.lt/kiosk
echo.
echo Note: If the tunnel prompts for an IP/password on your phone, 
echo just click the "Click to Continue" button on the webpage.
echo ========================================================
echo.

npx localtunnel --port 8001 --subdomain isl-banking
