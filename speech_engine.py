""" Speech Engine Module for recognizing speech and converting it to text. """
from queue import Queue
import os
import threading
import logging
import speech_recognition as sr
from utils.useful_funcs import bcolors, silence

logger = logging.getLogger(__name__)

class SpeechController:
    """
    Speech Controller for recognizing speech and converting it to text.
    """
    def __init__(self,
                 tts_engine,
                 command_handler,
                 recog_type: str = "sphinx",
                 pause_threshold: int = 2,
                 ambient_duration: int = 5) -> None:
        self.tts_engine = tts_engine
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=16000)
        self.recog_type: str = recog_type
        self.command_handler = command_handler
        self.stop_listening = None
        self.recognized_text_queue = Queue()
        self.timer_thread = None
        self.waiting_for_query = threading.Event()
        self.waiting_for_query.clear()

        self.recognizer.pause_threshold = pause_threshold
        with self.microphone as source:
            print(f"{bcolors.HEADER}Calibrating microphone for {ambient_duration} seconds...{bcolors.ENDC}")
            self.recognizer.adjust_for_ambient_noise(source, duration=ambient_duration)
            print(f"{bcolors.OKGREEN}Calibrated microphone.{bcolors.ENDC}\n")
        
        print(f"{bcolors.OKGREEN}Speech recognition initialized with {recog_type} engine.{bcolors.ENDC}\n")

    def background_listen(self, recognizer, audio) -> None:
        """
        Background listener callback for speech recognition.

        Parameters
        ----------
        recognizer : speech_recognition.Recognizer
            Recognizer object for speech recognition.
        audio : speech_recognition.AudioData
            Audio data recorded by the microphone from background listener.
        """
        if self.recognized_text_queue.empty() and not self.waiting_for_query.is_set():
            try:
                recognized_text = recognizer.recognize_sphinx(audio, keyword_entries=[("hey computer", 0.2)]).lower()
                unique_tokens = set(recognized_text.split(" "))
                if "hey" in unique_tokens and "computer" in unique_tokens:
                    self.tts_engine.speak("Yes? What do you want?")
                    print("Waiting for query...")
                    self.waiting_for_query.set()
            except sr.UnknownValueError as _:
                pass
            except Exception as generic_exception: # pylint: disable=bare-except,broad-exception-caught
                raise generic_exception

    def run_background(self, stop_time: int = 0) -> None:
        """ Run the background listener for speech recognition. """
        print(f"{bcolors.OKGREEN}Beginning background speech recognition.{bcolors.ENDC}\n")
        print(f"{bcolors.HEADER}Listening for keyword...{bcolors.ENDC}")
        if not self.stop_listening:
            self.stop_listening = self.recognizer.listen_in_background(
                self.microphone, self.background_listen, phrase_time_limit=3
            )
        # create a new thread to run the background listener
        thread = threading.Thread(target=self.run_foreground, args=[stop_time])
        thread.start()

    def run_foreground(self, stop_time: int) -> None:
        """ Run the foreground listener for speech recognition. """
        if stop_time > 0:
            self.timer_thread = threading.Timer(stop_time, self.stop_background)
            self.timer_thread.daemon = True
            self.timer_thread.start()
        while self.is_running():
            self.waiting_for_query.wait()
            try:
                audio = self.recognizer.listen(self.microphone, phrase_time_limit=5)
                # Requires Google Speech Recognition API key, otherwise, uses default
                if self.recog_type == "google":
                    with silence():
                        if not os.environ.get("GOOGLE_API_KEY"):
                            recognized_text = self.recognizer.recognize_google(audio)
                        else:
                            recognized_text = self.recognizer.recognize_google(audio, key=os.environ.get("GOOGLE_API_KEY"))
                # Requires whisper library
                elif self.recog_type == "whisper":
                    recognized_text =  self.recognizer.recognize_whisper(audio, language="english", model="medium.en")
                # Requires pocketsphinx library
                elif self.recog_type == "sphinx":
                    recognized_text = self.recognizer.recognize_sphinx(audio, language="en-US")
                elif self.recog_type == "wit":
                    if not os.environ.get("WIT_AI_KEY"):
                        raise ValueError("Wit.ai key not found in environment variables, please set WIT_AI_KEY or use different recognizer type.")
                    recognized_text = self.recognizer.recognize_wit(audio, key=os.environ.get("WIT_AI_KEY"))
                else:
                    recognized_text = "Invalid recognition type."

                if recognized_text != "":
                    print(f"{bcolors.HEADER}Recognized text:{bcolors.ENDC} {bcolors.OKCYAN}{recognized_text}{bcolors.ENDC}")
                    self.recognized_text_queue.put(recognized_text)
                    command = self.recognized_text_queue.get()
                    self.command_handler.handle_command(command, self.timer_thread)
                    self.recognized_text_queue.task_done()
                    self.waiting_for_query.clear()
                    print(f"\n{bcolors.HEADER}Listening for keyword...{bcolors.ENDC}")
            except sr.UnknownValueError:
                pass
            except sr.RequestError as request_error:
                print(f"{bcolors.FAIL}Could not request results from Speech Recognition service; {request_error}{bcolors.ENDC}")

    def stop_background(self) -> None:
        """ Stop the background listener for speech recognition. """
        print(f"{bcolors.OKGREEN}Stopping background speech recognition.{bcolors.ENDC}\n")
        self.stop_listening = self.stop_listening(wait_for_stop=False)

    def is_running(self) -> bool:
        """ Check if the background listener is running. """
        return self.stop_listening is not None
