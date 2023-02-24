""" Command to read the current time. """
from datetime import datetime
from commands.command import Command


class ReadTimeCommand(Command):
    """ This class is the command to read the current time. """
    def __init__(self) -> None:
        self.command_aliases = ['time']
        super().__init__(command_aliases = self.command_aliases, intent = "wit_time", requires_args = False)

    def execute_string_command(self, _) -> str:
        current_time = datetime.now()
        day_ordinal = self.ordinal(current_time.day)
        greeting = self.greeting(current_time.hour)
        time_string = current_time.strftime(f'{greeting}! It is %-I:%M %p on %A of %B {day_ordinal}, %Y')
        return time_string

    def execute_intent_command(self, _) -> str:
        current_time = datetime.now()
        day_ordinal = self.ordinal(current_time.day)
        greeting = self.greeting(current_time.hour)
        time_string = current_time.strftime(f'{greeting}! It is %-I:%M %p on %A of %B {day_ordinal}, %Y')
        return time_string

    def ordinal(self, day: int) -> str:
        """
        Derive the ordinal numeral for a given number n
        Taken from: https://stackoverflow.com/questions/66364289/
        """
        return f"{day:d}{'tsnrhtdd'[(day//10%10!=1)*(day%10<4)*day%10::4]}"

    def greeting(self, hour: int) -> str:
        """
        Return a greeting based on the current time
        """
        if 0 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"
