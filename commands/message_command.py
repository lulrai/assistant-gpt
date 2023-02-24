""" This module contains the MessageCommand class. """
import json
import os
from sys import platform
from commands.command import Command


class MessageCommand(Command):
    """ This class is the command to send a message. """
    def __init__(self) -> None:
        self.command_aliases = ['message', 'text', 'sms', 'send']
        super().__init__(command_aliases = self.command_aliases, intent="wit_message", requires_args = True)
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone_number.json"), "r", encoding="utf-8") as num_json:
                self.possible_numbers = json.load(num_json)
                if "default" not in self.possible_numbers:
                    self.error_flag = True
        except FileNotFoundError:
            self.error_flag = True
            
    def execute_string_command(self, args) -> str:
        """ This method executes the command and returns the message. """
        # Check if the command is available on the current platform
        if platform != "darwin":
            return "This command is only available on macOS."

        # Check if the command has issues
        if self.error_flag:
            return "Either the phone number file is missing or the default number is not set."

        # Remove the command aliases from the arguments
        for alias in self.command_aliases:
            if alias in args:
                args.remove(alias)
        target_number = self.possible_numbers["default"]
        for name, number in self.possible_numbers.items():
            if name in args:
                args.remove(name)
                target_number = number
        message = " ".join(args)

        # Send the message
        response_message = ""
        # Open Messages app and send a message to a contact
        if os.system(f'''/usr/bin/osascript -e '{self.generate_applescript(target_number=target_number, message=message)}' ''') == 0:
            if os.system('''/usr/bin/osascript -e 'tell application "Messages" to quit' ''') == 0:
                response_message = "Message sent!"
            else:
                response_message = "Message sent, but failed to close Messages app."
        else:
            response_message = "Could not send message!"
        return response_message

    def execute_intent_command(self, args) -> str:
        # Check if the command is available on the current platform
        if platform != "darwin":
            return "This command is only available on macOS."

        # Check if the command has issues
        if self.error_flag:
            return "Either the phone number file is missing or the default number is not set."

        print(args)
        return "This command is not yet implemented."

    def generate_applescript(self, target_number: str, message: str) -> str:
        """ This method generates the AppleScript to send a message. """
        applescript = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            set targetBuddy to buddy "{target_number}" of targetService
            set textMessage to "{message}"
            send textMessage to targetBuddy
        end tell'''
        return applescript
