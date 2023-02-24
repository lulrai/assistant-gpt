""" Command Handler Module to handle commands. """
import json
import os
import sys
import threading
import importlib
from datetime import datetime
from typing import Union

import requests
from commands.command import Command
from utils.useful_funcs import bcolors


class CommandHandler:
    """ This class initializes and handles commands. """
    def __init__(self, tts_engine):
        self.commands = {}
        self.tts_engine = tts_engine
        self.load_commands()

    def load_commands(self):
        """
        Dynamically loads all Command classes in the commands directory
        """
        print(f"{bcolors.HEADER}Loading commands...{bcolors.ENDC}")
        commands_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands")
        file_count = 0
        for filename in os.listdir(commands_dir):
            if filename.endswith(".py") and not filename.startswith("__") and filename != "command.py":
                module_name = "commands." + filename[:-3]
                module = importlib.import_module(module_name)
                for _, obj in vars(module).items():
                    if isinstance(obj, type) and issubclass(obj, Command) and obj is not Command:
                        command_instance = obj()
                        file_count += 1
                        for command_alias in command_instance.get_command_aliases():
                            if command_alias not in self.commands:
                                self.commands[command_alias] = {}
                                self.commands[command_alias]["requires_args"] = command_instance.arg_required()
                                self.commands[command_alias]["string_command"] = command_instance.execute_string_command
                                self.commands[command_alias]["intent_command"] = command_instance.execute_intent_command
                            else:
                                raise ValueError(f"Command alias {command_alias} already exists.")
                        print(f"{bcolors.OKBLUE}Loaded command {obj.__name__} with aliases {command_instance.command_aliases}.{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Loaded {file_count} commands with {len(self.commands)} aliases.{bcolors.ENDC}\n")

    def handle_command(self, command: Union[str, dict], timer_thread: threading.Thread) -> None:
        """
        Handles the command by executing the command if it exists, depending on the type of command.
        """
        # If the command is a dict, it is an intent command from wit.ai
        if isinstance(command, dict):
            self.__handle_intent_command(command, timer_thread)
        # If the command is a string, it is a string command
        elif isinstance(command, str):
            self.__handle_string_command(command, timer_thread)
        # If the command is neither, raise a TypeError
        else:
            raise TypeError(f"Command must be a string or dict, not {type(command)}.")

    def __handle_string_command(self, command: str, timer_thread: threading.Thread) -> str:
        """ Handles the command if it is a string. """
        command = command.lower()
        returned_response = ""
        if command.strip('.,').replace(' ', '') in self.commands:
            if self.commands[command.strip('.,').replace(' ', '')]["requires_args"]:
                returned_response = "Command requires arguments."
            else:
                returned_response = self.commands[command.strip('.,').replace(' ', '')]["string_command"]()
        else:
            for command_alias, command_info in self.commands.items():
                args = command.lower().split(" ")
                args = [arg.strip('.,') for arg in args]
                if command_alias in args:
                    if command_info["requires_args"]:
                        args.remove(command_alias)
                        returned_response = command_info["function"](args)
                    else:
                        returned_response = command_info["function"]()
                    break
        if returned_response == "":
            returned_response = "Sorry, command not found."
        print(f"{bcolors.HEADER}Response:{bcolors.ENDC} {bcolors.OKGREEN}{returned_response}{bcolors.ENDC}")
        self.tts_engine.speak(returned_response)
        if "goodbye" in returned_response.lower():
            if timer_thread:
                timer_thread.cancel()
            sys.exit()

    def __handle_intent_command(self, command_response: dict, timer_thread: threading.Thread) -> str:
        """ Handles the command if it is an intent command. """
        entities = command_response["entities"]
        intent = command_response["intents"][0]['name']
        returned_response = ""
        
        for _, command_info in self.commands.items():
            if command_info.get("intent_command") == intent:
                if command_info["requires_args"]:
                    returned_response = command_info["intent_command"](entities)
                else:
                    returned_response = command_info["intent_command"]()
                break
        if returned_response == "":
            returned_response = "Sorry, command not found."
        print(f"{bcolors.HEADER}Response:{bcolors.ENDC} {bcolors.OKGREEN}{returned_response}{bcolors.ENDC}")
        self.tts_engine.speak(returned_response)
        if "goodbye" in returned_response.lower():
            if timer_thread:
                timer_thread.cancel()
            sys.exit()


    def learn_wit(self) -> bool:
        """ Learns wit.ai intents and utterances. """
        # Get the most recent version of the API
        api_version = datetime.now().strftime("%Y%m%d")

        # URLs for all the requests
        utterance_url = f"https://api.wit.ai/utterances?v={api_version}"
        intent_url = f"https://api.wit.ai/intents?v={api_version}"
        entity_url = f"https://api.wit.ai/entities?v={api_version}"

        # headers for all the requests
        headers = {'Authorization': f'Bearer {os.environ.get("WIT_AI_KEY")}', 'Content-Type': 'application/json'}

        # Directory where all the command utterances are stored
        witai_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "witai")
        if not os.path.exists(witai_dir):
            print(f"{bcolors.FAIL}witai directory does not exist.{bcolors.ENDC}")
            return False

        print(f"{bcolors.OKCYAN}Learning wit.ai intents and entities...{bcolors.ENDC}")
        # Create a requests session
        with requests.Session() as session:
            # POST all the intents and entities
            with open(os.path.join(witai_dir, "intent_entity_setup.json"), "r", encoding="utf-8") as setup_file:
                setup = json.load(setup_file)

                # POST intents
                for intent in setup["intents"]:
                    data = { "name": intent }
                    response = session.post(intent_url, headers=headers, data=json.dumps(data), timeout=5)
                    if response.json().get("error") and "already exists" in response.json().get("error"):
                        continue
                    if response.status_code != 201 and response.status_code != 200:
                        print(f"{bcolors.FAIL}Failed to learn wit.ai intent {intent}.{bcolors.ENDC}")
                        return False

                # POST entities
                for entity in setup["entities"]:
                    data = { "name": entity[0], "roles": [], "lookups": entity[1] }
                    response = session.post(entity_url, headers=headers, data=json.dumps(data), timeout=5)
                    if response.json().get("error") and "already exists" in response.json().get("error"):
                        continue
                    if response.status_code != 201 and response.status_code != 200:
                        print(f"{bcolors.FAIL}Failed to learn wit.ai entity {entity[0]}.{bcolors.ENDC}")
                        return False
            print(f"{bcolors.OKBLUE}Done learning wit.ai intents and entities.{bcolors.ENDC}")
            print(f"{bcolors.OKCYAN}Learning wit.ai command utterances...{bcolors.ENDC}")
            # POST command utterances
            for filename in os.listdir(os.path.join(witai_dir, "command_utterances")):
                if filename.endswith(".json"):
                    with open(os.path.join(witai_dir, "command_utterances", filename), "r+", encoding="utf-8") as utterance_file:
                        utterances = json.load(utterance_file)
                        for utterance_text, utterance_info in utterances.items():
                            if utterance_info.get("done") is None:
                                data = {
                                    "text": utterance_text,
                                    "intent": utterance_info["intent"],
                                    "entities": utterance_info["entities"],
                                }
                                response = session.post(utterance_url, headers=headers, data=json.dumps(data), timeout=5)
                                print(response.text)
                                if response.status_code != 201 and response.status_code != 200:
                                    print(f"{bcolors.FAIL}Failed to learn wit.ai utterance {utterance_text}.{bcolors.ENDC}")
                                    return False
                                utterance_info["done"] = True
                        # utterance_file.seek(0)
                        # json.dump(utterances, utterance_file, indent=4)
                        # utterance_file.truncate()
            print(f"{bcolors.OKBLUE}Done learning wit.ai command utterances.{bcolors.ENDC}")
        return True
