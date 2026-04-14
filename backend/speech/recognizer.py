
import os
import sys
import json
import wave
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VOSK_MODEL_PATH, VOSK_SAMPLE_RATE


class VoskRecognizer:
    """
    Offline speech-to-text using Vosk.

    Captures audio from the microphone using PyAudio, processes it
    through a Vosk model, and returns the recognized text.

    Attributes:
        model: Vosk Model instance.
        recognizer: Vosk KaldiRecognizer instance.
        sample_rate: Audio sample rate (default: 16000).
    """

    def __init__(self, model_path: Optional[str] = None,
                 sample_rate: int = VOSK_SAMPLE_RATE):
        """
        Initialize the Vosk recognizer.

        Args:
            model_path: Path to the Vosk model directory.
                        If None, uses the default path from config.
            sample_rate: Audio sample rate in Hz (default: 16000).
        """
        self.sample_rate = sample_rate
        self.model = None
        self.recognizer = None
        self._initialized = False

        path = Path(model_path) if model_path else VOSK_MODEL_PATH

        if not path.exists():
            logger.warning(
                f"Vosk model not found at {path}. "
                f"Download from https://alphacephei.com/vosk/models "
                f"and extract to {path}"
            )
            return

        try:
            from vosk import Model, KaldiRecognizer
            self.model = Model(str(path))
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            self._initialized = True
            logger.info(f"Vosk recognizer initialized with model: {path}")
        except ImportError:
            logger.warning("Vosk not installed. Run: pip install vosk")
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if the recognizer is properly initialized."""
        return self._initialized

    def listen(self, duration: float = 5.0) -> Optional[str]:
        """
        Listen to microphone and transcribe speech.

        Captures audio for the specified duration and returns
        the recognized text.

        Args:
            duration: Recording duration in seconds (default: 5.0).

        Returns:
            Recognized text string, or None if failed.
        """
        if not self._initialized:
            logger.error("Vosk recognizer not initialized")
            return None

        try:
            import pyaudio

            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=8000
            )

            logger.info(f"Listening for {duration}s...")
            frames_to_read = int(self.sample_rate * duration / 8000)

            for _ in range(frames_to_read):
                data = stream.read(8000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        return result['text']

            # Get final result
            final = json.loads(self.recognizer.FinalResult())
            stream.stop_stream()
            stream.close()
            p.terminate()

            return final.get('text', '')

        except ImportError:
            logger.error("PyAudio not installed. Run: pip install pyaudio")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        """
        Transcribe an audio file (WAV format).

        Args:
            audio_path: Path to a WAV audio file.

        Returns:
            Recognized text string, or None if failed.
        """
        if not self._initialized:
            logger.error("Vosk recognizer not initialized")
            return None

        try:
            wf = wave.open(audio_path, 'rb')

            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                logger.error("Audio must be mono 16-bit WAV")
                return None

            from vosk import KaldiRecognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    partial = json.loads(rec.Result())
                    if partial.get('text'):
                        results.append(partial['text'])

            final = json.loads(rec.FinalResult())
            if final.get('text'):
                results.append(final['text'])

            wf.close()
            return ' '.join(results)

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return None
