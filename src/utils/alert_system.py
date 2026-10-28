"""
Alert system module for Driver Monitoring System.
Provides audio buzzer/siren (pygame / winsound) and speech alerts (pyttsx3).
Runs asynchronously in a background worker thread with non-blocking queue.
"""

import sys
import threading
import queue
import time
from typing import Optional, Dict
import numpy as np
import src.config as config


class AudioSynthesizer:
    """Generates synthetic sounds without external audio files using pygame.sndarray."""
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sounds: Dict[str, any] = {}
        self.is_ready = False
        self._init_sounds()

    def _init_sounds(self):
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate)

            # 1. Loud Siren (for Sleep / Drowsy)
            d1 = 0.35
            t1 = np.linspace(0, d1, int(self.sample_rate * d1), endpoint=False)
            wave1 = 0.45 * np.sin(2 * np.pi * 2600 * t1)
            mono1 = np.int16(wave1 * 32767)
            stereo1 = np.column_stack((mono1, mono1))
            self.sounds["SIREN_LOUD"] = pygame.sndarray.make_sound(stereo1)

            # 2. Rapid Siren (for Hands-off Steering Wheel)
            d2 = 0.15
            t2 = np.linspace(0, d2, int(self.sample_rate * d2), endpoint=False)
            wave2 = 0.50 * np.sin(2 * np.pi * 3200 * t2)
            mono2 = np.int16(wave2 * 32767)
            stereo2 = np.column_stack((mono2, mono2))
            self.sounds["SIREN_RAPID"] = pygame.sndarray.make_sound(stereo2)

            # 3. Soft Chime (for Yawning / Reminder)
            d3 = 0.25
            t3 = np.linspace(0, d3, int(self.sample_rate * d3), endpoint=False)
            # Harmonic chime: 880Hz + 1760Hz with decay
            decay = np.exp(-3.0 * t3)
            wave3 = 0.35 * (np.sin(2 * np.pi * 880 * t3) + 0.5 * np.sin(2 * np.pi * 1760 * t3)) * decay
            mono3 = np.int16(wave3 * 32767)
            stereo3 = np.column_stack((mono3, mono3))
            self.sounds["CHIME"] = pygame.sndarray.make_sound(stereo3)

            self.is_ready = True
        except Exception as e:
            # Pygame audio synthesis not available
            self.is_ready = False

    def play(self, sound_key: str):
        if self.is_ready and sound_key in self.sounds:
            try:
                self.sounds[sound_key].play()
                return True
            except Exception:
                return False
        return False


class AlertSystem:
    def __init__(self, enable_audio: bool = config.ENABLE_AUDIO):
        self.enable_audio = enable_audio
        self.audio_queue = queue.Queue(maxsize=15)
        self.is_running = True
        self.synthesizer = None
        self.tts_engine = None

        if self.enable_audio:
            self._init_drivers()

        # Worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _init_drivers(self):
        """Initialize pygame synthesizer and pyttsx3 safely."""
        self.synthesizer = AudioSynthesizer()

        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', config.VOICE_RATE)
            self.tts_engine.setProperty('volume', config.VOICE_VOLUME)
        except Exception:
            self.tts_engine = None

    def trigger(self, action: str, voice_msg: Optional[str] = None):
        """
        Queue an alert action.
        - action: 'SIREN_LOUD' | 'SIREN_RAPID' | 'CHIME' | 'VOICE'
        - voice_msg: Optional spoken message string
        """
        if not self.enable_audio:
            return
        try:
            self.audio_queue.put_nowait((action, voice_msg))
        except queue.Full:
            pass

    def _worker_loop(self):
        """Consume and execute audio commands asynchronously."""
        while self.is_running:
            try:
                action, voice_msg = self.audio_queue.get(timeout=0.2)

                # 1. Play sound effect
                played = False
                if self.synthesizer and action in ("SIREN_LOUD", "SIREN_RAPID", "CHIME"):
                    played = self.synthesizer.play(action)

                # Fallback to winsound on Windows if synthesizer failed
                if not played and action in ("SIREN_LOUD", "SIREN_RAPID", "CHIME"):
                    if sys.platform == "win32":
                        try:
                            import winsound
                            if action == "SIREN_LOUD":
                                winsound.Beep(2600, 350)
                            elif action == "SIREN_RAPID":
                                winsound.Beep(3200, 150)
                                winsound.Beep(3200, 150)
                            elif action == "CHIME":
                                winsound.Beep(1200, 200)
                        except Exception:
                            pass

                # 2. Speak voice message
                if voice_msg and self.tts_engine is not None:
                    try:
                        self.tts_engine.say(voice_msg)
                        self.tts_engine.runAndWait()
                    except Exception:
                        pass

                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.05)

    def stop(self):
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
