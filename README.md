# Sign Language Kiosk
Real-time, bidirectional assistive communication system that allows Deaf users to
communicate with bank employees using Indian Sign Language (ISL), without
requiring a human interpreter.

All processing is local and privacy-preserving — raw video frames are discarded
immediately after landmark extraction; only mathematical coordinates are stored.

# & "C:\Users\LENOVO\Desktop\Minor MVP\isl_env\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000