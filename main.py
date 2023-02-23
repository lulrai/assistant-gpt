""" Main file for the project. """
from dotenv import load_dotenv
from tts_engine import TextToSpeechEngine
from speech_engine import SpeechController
from command_handler import CommandHandler


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    # Create a new tts object
    tts_engine = TextToSpeechEngine(tts_type = "ai")

    # Create a new command handler
    command_handler = CommandHandler(tts_engine = tts_engine)

    # Create a new speech controller
    speech_controller = SpeechController(
        tts_engine = tts_engine, recog_type = "google", command_handler = command_handler
    )

    # Run the speech controller in the background
    speech_controller.run_background()
