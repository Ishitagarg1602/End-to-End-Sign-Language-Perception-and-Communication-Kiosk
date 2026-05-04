import os
import shutil

src = r"d:\GitHub\End-to-End-Sign-Language-Perception-and-Communication-Kiosk\frontend\src\assets\MAN.mp4"
dst_dir = r"d:\GitHub\End-to-End-Sign-Language-Perception-and-Communication-Kiosk\frontend\public"
dst = os.path.join(dst_dir, "MAN.mp4")

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

if os.path.exists(src):
    shutil.move(src, dst)
    print(f"Moved {src} to {dst}")
else:
    print(f"Source file {src} does not exist. It might have already been moved.")
