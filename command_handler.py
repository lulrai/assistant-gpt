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
        api_version = datetime.now().strftime("%Y%m%d")
        url = f"https://api.wit.ai/utterances?v={api_version}"
        # make a post request to wit.ai to learn the command
        headers = {'Authorization': f'Bearer {os.environ.get("WIT_AI_KEY")}', 'Content-Type': 'application/json'}

        commands_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands", "command_utterances")
        if not os.path.exists(commands_dir):
            return False
        for filename in os.listdir(commands_dir):
            if filename.endswith(".json"):
                with open(os.path.join(commands_dir, filename), "w", encoding="utf-8") as utterance_file:
                    utterances = json.load(utterance_file)
                    for utterance in utterances:
                        if utterance.get("done") is None:
                            data = {
                                "text": utterance["text"],
                                "intent": utterance["intent"],
                                "entities": utterance["entities"],
                            }
                            try:
                                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=5)
                                if response.status_code != 201:
                                    return False
                            except requests.exceptions.Timeout:
                                return False
                            utterance["done"] = True
                    json.dump(utterances, utterance_file, indent=4)
        return True
