""" This module contains the Command class. """
from abc import ABC, abstractmethod


class Command(ABC):
    """ This class is the base class for all commands. """
    def __init__(self, command_aliases: list, intent: str, requires_args: bool) -> None:
        super().__init__()
        self.command_aliases: list = command_aliases
        self.requires_args: bool = requires_args
        self.intent: str = intent

    @abstractmethod
    def execute_string_command(self, args) -> str:
        """ This method executes the string command. """

    @abstractmethod
    def execute_intent_command(self, args) -> str:
        """ This method executes the intent command. """

    def get_command_aliases(self) -> list:
        """ This property returns the command aliases. """
        return self.command_aliases

    def arg_required(self) -> bool:
        """ This property returns whether the command requires arguments. """
        return self.requires_args

    def get_intent(self) -> str:
        """ This property returns the intent of the command. """
        return self.intent
