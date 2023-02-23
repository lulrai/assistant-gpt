""" This module contains the ExitCommand class. """""
from commands.command import Command


class ExitCommand(Command):
    """ This class is the command to exit the program. """
    def __init__(self):
        self.command_aliases = ['abrakadabra', 'abracadabra', 'goodbye', 'goodnight']
        super().__init__(command_aliases = self.command_aliases, requires_args = False)

    def execute_command(self, *_) -> str:
        """ This method executes the command. """
        return "Goodbye!"
