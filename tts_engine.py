""" Text to speech engine for the project. """
import time
import pyttsx3
import pyaudio
import numpy as np
from TTS.api import TTS
from utils.useful_funcs import silence, bcolors


class TextToSpeechEngine:
    """ Text to speech engine for the project. """
    def __init__(self, tts_type: str = "default", tts_rate: int = 170, tts_volume: float = 1.0) -> None:
        self.tts_type: str = tts_type
        self.tts_rate: int = tts_rate
        self.tts_volume: int = tts_volume
        print(f"{bcolors.HEADER}Initializing {self.tts_type.upper()} TTS engine...{bcolors.ENDC}")
        if self.tts_type == "ai":
            with silence():
                self.tts = TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=False)
        else:
            self.speaker = pyttsx3.init()
        print(f"{bcolors.OKGREEN}TTS engine initialized.{bcolors.ENDC}\n")

    def __syllable_count(self, phrase: str) -> int:
        vowels = "aeiouy"
        phrase = phrase.lower()
        count = sum(1 for letter in phrase if letter in vowels)
        return count

    def speak(self, text: str) -> None:
        """ Given a string, speak it out loud. """
        if self.tts_type == "ai":
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=22050,
                frames_per_buffer=1024,
                output=True,
                output_device_index=2
            )
            try:
                with silence():
                    samples = self.tts.tts(text=text, speaker=self.tts.speakers[2])
                samples = np.array(samples, dtype=np.float32)
                total_time = len(samples) / 22050
                stream.write(samples.tobytes())
            finally:
                stream.close()
                p.terminate()
        else:
            self.speaker = pyttsx3.init()
            try:
                self.speaker.endLoop()
                del self.speaker
                self.speaker = pyttsx3.init()
            except: # pylint: disable=bare-except
                pass
            self.speaker.startLoop()
            self.speaker.setProperty('rate', self.tts_rate)     # setting up new voice rate
            self.speaker.setProperty('volume', self.tts_volume)    # setting up volume level  between 0 and 1
            self.speaker.say(text)
            total_time = (self.__syllable_count(text) / self.tts_rate) * 60
            time.sleep(total_time)
