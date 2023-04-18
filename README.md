<a name="readme-top"></a>

<section align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

</section>

<section align="center">

<p></p>

[![LinkedIn][linkedin-shield]][linkedin-url]

</section>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <picture>
      <source 
      srcset="resources/images/logo_dark.png"
      media="(prefers-color-scheme: dark), (prefers-color-scheme: no-preference)"
      />
      <source
      srcset="resources/images/logo_light.png"
      media="(prefers-color-scheme: light)"
      />
      <img src="resources/images/logo_dark.png" />
  </picture>

  <h3 align="center">AssistantGPT</h3>

  <p align="center">
    A fun speech-to-text and text-to-speech based assistant for your needs.
    <br />
    <a href="https://github.com/lulrai/assistant-gpt/issues">Report Bug</a> 
    |
    <a href="https://github.com/lulrai/assistant-gpt/issues">Request Features</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#features">Features</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

<section align="center">

[![Application ScreenShot][product-screenshot]]()

</section>

Assistant GPT (APT) is a Python application that uses speech recognition to allow users to interact with their computer using voice commands. It supports multiple recognition engines and provides a variety of built-in commands as well as the ability to define custom commands.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

The project was built with `Python >= 3.8`. All the dependencies are listed below:

- [![Conda][conda]][conda-url]
- [![Python][python]][python-url]
- [![Google][google]][google-url]
- [![RevChatGPT][revchatgpt]](revchatgpt-url)
- [![PyTorch][pytorch]](pytorch)
- [![SpeechRecognition][speechrecog]](speechrecog-url)
- [![Coqui-Ai-TTS][tts]](tts-url)
- [![Open-Ai-Whisper][whisper]](whisper-url)
- [![PocketSphinx][pocketsphinx]](pocketsphinx-url)
- [![Wit.Ai][witai]](witai-url)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

Always expanding features but here are a few of the current features present in the application.
- Support for multiple speech recognition engines, including Google, Sphinx, Whisper, and Wit.ai.
- Calibrate microphone to adjust for ambient noise before speech recognition.
- Background listener for voice-activated commands with phrase time limit.
- Foreground listener for speech recognition with phrase time limit.
- Custom command creation with customizable voice triggers and actions.
- Text-to-speech (TTS) engine integration for computer responses.
- Interactive mode for continuous conversation with the computer.
- (Optional) Automatic stop time for background listener.
- Currently implemented commands:
    - _TimeCommand_:
        > Triggers: time

        Ask the current time which will reply both current time and date.
    
    - _ChatGPTCommand_:
        > Triggers: chatgpt, gpt, ask

        Ask ChatGPT questions and it will reply back.
    
    - _ExitCommand_:
        > Triggers: goodbye, goodnight, abracadabra, abrakadabra

        Stops listening to further commands and terminates the program.
    
    - _(Experimental) MessageCommand_: (**NOTE: ONLY WORKS ON MACOS**) 
        > Triggers: message, text, sms, send

        Sends a message using the iMessage app on MacOS to the person you ask. 

## Usage

Coming soon!
<!-- This is just version 1 of the website so still a work in progress but to run this, the steps are as follows:

#### Setup

- Install `yarn` and `npm`.
- Install `firebase` and `vite`.
- Clone the repository, obviously.

#### Running

- Open the folder in an editor like **VSCode**.
- Install all dependencies required using the following command:

```bash
yarn
```

- (Optional) Login to firebase and setup the project, follow the steps given by the command:

```bash
firebase login
```

- Finally, run the command to host the server:

```bash
yarn dev
``` -->

<!-- _Be Aware: There will be various changes and updates will be pushed very often._ -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [ ] Add further support for more text-to-speech and speech-to-text engines
- [ ] Add more commands such as:
    - [ ] Opening applications
    - [ ] Controlling keyboard using pyautogui
    - And more!
- [ ] Automate retrieving ```access_token``` for ChatGPT using user login.
- [ ] Automatically generate ```.env``` variables but keep it empty.
- [ ] Make this project a distributable library/package?

See the [open issues](https://github.com/lulrai/assistant-gpt/issues) for a full list of known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Project Link: [https://github.com/lulrai/assistant-gpt](https://github.com/lulrai/assistant-gpt)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Img Shields](https://shields.io)
- [List of All Image Shields](https://github.com/progfay/shields-with-icon)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/lulrai/assistant-gpt.svg?style=for-the-badge
[contributors-url]: https://github.com/lulrai/assistant-gpt/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lulrai/assistant-gpt.svg?style=for-the-badge
[forks-url]: https://github.com/lulrai/assistant-gpt/network/members
[stars-shield]: https://img.shields.io/github/stars/lulrai/assistant-gpt.svg?style=for-the-badge
[stars-url]: https://github.com/lulrai/assistant-gpt/stargazers
[issues-shield]: https://img.shields.io/github/issues/lulrai/assistant-gpt.svg?style=for-the-badge
[issues-url]: https://github.com/lulrai/assistant-gpt/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-blue?style=for-the-badge&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/nneupane2/
[linkedin-url]: https://www.linkedin.com/in/nneupane2/
[product-screenshot]: resources/images/product_screenshot.png
[conda]: https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white
[conda-url]: https://docs.conda.io/en/latest/
[python]: https://img.shields.io/static/v1?style=for-the-badge&message=Python&color=3776AB&logo=Python&logoColor=FFFFFF&label=3.9
[python-url]: https://www.python.org
[google]: https://img.shields.io/static/v1?style=for-the-badge&message=Google+Speech+Recognition&color=4285F4&logo=Google+Cloud&logoColor=FFFFFF&label=
[google-url]: https://cloud.google.com/speech-to-text
[revchatgpt]: https://img.shields.io/badge/RevChatGPT-2.3-brightgreen.svg?style=for-the-badge
[revchatgpt-url]: https://github.com/acheong08/ChatGPT
[speechrecog]: https://img.shields.io/badge/SpeechRecognition-3.9.0-brightgreen.svg?style=for-the-badge
[speechrecog-url]: https://github.com/Uberi/speech_recognition
[tts]: https://img.shields.io/badge/TTS-0.11.1-brightgreen.svg?style=for-the-badge
[tts-url]: https://github.com/coqui-ai/TTS
[whisper]: https://img.shields.io/badge/Whisper-20230124-brightgreen.svg?style=for-the-badge
[whisper-url]: https://github.com/openai/whisper
[pocketsphinx]: https://img.shields.io/badge/PocketSphinx-5.0.0-brightgreen.svg?style=for-the-badge
[pocketsphinx-url]: https://github.com/cmusphinx/pocketsphinx
[pytorch]: https://img.shields.io/static/v1?style=for-the-badge&message=PyTorch&color=EE4C2C&logo=PyTorch&logoColor=FFFFFF&label=
[pytorch-url]: https://pytorch.org/get-started/locally/
[witai]: https://img.shields.io/badge/Wit.Ai-brightgreen.svg?style=for-the-badge
[witai-url]: https://github.com/wit-ai
