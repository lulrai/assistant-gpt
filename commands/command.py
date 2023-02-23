""" This module contains the Command class. """
from abc import ABC, abstractmethod


class Command(ABC):
    """ This class is the base class for all commands. """
    def __init__(self, command_aliases: list, requires_args: bool) -> None:
        super().__init__()
        self.command_aliases: list = command_aliases
        self.requires_args: bool = requires_args

    @abstractmethod
    def execute_command(self, args) -> str:
        """ This method executes the command. """

    def get_command_aliases(self) -> list:
        """ This property returns the command aliases. """
        return self.command_aliases

    def arg_required(self) -> bool:
        """ This property returns whether the command requires arguments. """
        return self.requires_args
