
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2


# ─── Constants ───────────────────────────────────────────────────────────────
DATASET_DIR = Path('./mvp/dataset')
CAMERA_INDEX = 0
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_DURATION_SECONDS = 3
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')


def get_next_filename(word_dir: Path, word: str) -> str:
    """
    Generate the next available filename for a new recording.

    Follows pattern: <word>_new_r<N>.mp4

    Args:
        word_dir: Path to the word's dataset folder.
        word: The word being recorded.

    Returns:
        Next available filename string.
    """
    existing = list(word_dir.glob(f"{word}_new_r*.mp4"))
    next_num = len(existing) + 1
    return f"{word}_new_r{next_num}.mp4"


def record_video(word: str) -> None:
    """
    Record a single video for the given word using the webcam.

    Shows live preview. Press SPACE to start/stop recording.
    Press ESC to cancel.

    Args:
        word: The sign language word to record.
    """
    word_dir = DATASET_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)

    filename = get_next_filename(word_dir, word)
    output_path = word_dir / filename

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    writer = None
    recording = False
    frames_recorded = 0
    max_frames = FPS * MAX_DURATION_SECONDS

    print(f"\n  Camera opened. Preview is live.")
    print(f"  Press SPACE to start recording '{word}'")
    print(f"  Press SPACE again to stop (auto-stops after {MAX_DURATION_SECONDS}s)")
    print(f"  Press ESC to cancel\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from camera")
            break

        # Draw recording status overlay
        display = frame.copy()
        if recording:
            # Red recording indicator
            cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display, f"REC - {word.upper()} ({frames_recorded}/{max_frames})",
                       (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Ready indicator
            cv2.putText(display, f"READY - '{word}' | SPACE to record | ESC to exit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('ISL Dataset Recorder', display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            if recording and writer:
                writer.release()
                # Delete incomplete file
                if output_path.exists():
                    os.remove(str(output_path))
                    print("  Recording cancelled. File deleted.")
            break

        elif key == 32:  # SPACE
            if not recording:
                # Start recording
                writer = cv2.VideoWriter(str(output_path), FOURCC, FPS,
                                         (FRAME_WIDTH, FRAME_HEIGHT))
                recording = True
                frames_recorded = 0
                print(f"  ● Recording started → {filename}")
            else:
                # Stop recording
                recording = False
                if writer:
                    writer.release()
                print(f"  ■ Recording stopped. {frames_recorded} frames saved.")
                print(f"  Saved to: {output_path}")
                break

        # Write frame if recording
        if recording and writer:
            writer.write(frame)
            frames_recorded += 1

            # Auto-stop at max duration
            if frames_recorded >= max_frames:
                recording = False
                writer.release()
                print(f"  ■ Max duration reached. {frames_recorded} frames saved.")
                print(f"  Saved to: {output_path}")
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main entry point: interactive recording session.

    Loops asking for words to record until user types 'quit'.
    """
    print(f"{'='*60}")
    print(f"  ISL Dataset Recorder")
    print(f"{'='*60}")
    print(f"  Dataset directory: {DATASET_DIR}")
    print(f"  Camera index     : {CAMERA_INDEX}")
    print(f"  Resolution       : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  FPS              : {FPS}")
    print(f"  Max duration     : {MAX_DURATION_SECONDS}s")
    print(f"{'='*60}\n")

    while True:
        word = input("  Enter word to record (or 'quit' to exit): ").strip().lower()

        if word == 'quit' or word == 'exit' or word == 'q':
            print("  Goodbye!")
            break

        if not word:
            print("  [ERROR] Please enter a valid word.")
            continue

        # Show existing videos for this word
        word_dir = DATASET_DIR / word
        if word_dir.exists():
            existing = list(word_dir.glob("*.mp4"))
            print(f"  Existing videos for '{word}': {len(existing)}")
        else:
            print(f"  No existing videos for '{word}' — folder will be created.")

        record_video(word)
        print()


if __name__ == '__main__':
    main()
