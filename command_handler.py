""" Command Handler Module to handle commands. """
import os
import sys
import threading
import importlib
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
                                self.commands[command_alias]["function"] = command_instance.execute_command
                            else:
                                raise ValueError(f"Command alias {command_alias} already exists.")
                        print(f"{bcolors.OKBLUE}Loaded command {obj.__name__} with aliases {command_instance.command_aliases}.{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Loaded {file_count} commands with {len(self.commands)} aliases.{bcolors.ENDC}\n")

    def handle_command(self, command: str, timer_thread: threading.Thread) -> str:
        """
        Handles the command by executing the command if it exists.
        """
        command = command.lower()
        returned_response = ""
        if command.strip('.,').replace(' ', '') in self.commands:
            returned_response = self.commands[command.strip('.,').replace(' ', '')]["function"]()
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
        if "goodbye" in returned_response.lower():
            self.tts_engine.speak(returned_response)
            if timer_thread:
                timer_thread.cancel()
            sys.exit()
        self.tts_engine.speak(returned_response)