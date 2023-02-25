""" Command to read the current time. """
from datetime import datetime
from commands.command import Command


class ReadTimeCommand(Command):
    """ This class is the command to read the current time. """
    def __init__(self) -> None:
        self.command_aliases = ['time']
        super().__init__(command_aliases = self.command_aliases, intent = "wit_time", requires_args = True)

    def execute_string_command(self, args) -> str:
        current_time = datetime.now()
        day_ordinal = self.ordinal(current_time.day)
        greeting = self.greeting(current_time.hour)
        time_string = current_time.strftime(f'{greeting}! It is %-I:%M %p on %A of %B {day_ordinal}, %Y')
        return time_string

    def execute_intent_command(self, args) -> str:
        if args.get("timezone:timezone"):
            timezone = args["timezone:timezone"][0]["body"]
            print(self.__num_words_to_num(timezone))
            current_time = datetime.now()
            day_ordinal = self.ordinal(current_time.day)
            greeting = self.greeting(current_time.hour)
            time_string = current_time.strftime(f'{greeting}! It is %-I:%M %p on %A of %B {day_ordinal}, %Y')
        else:
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

    def __num_words_to_num(self, sentence: str) -> str:
        """ Convert a sentence with words to a sentence with numbers. """
        conv_dict = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "thirty": "30",
            "fourty-five": "45",
            "fourty five": "45",
            "plus": "+",
            "minus": "-",
            "gmt": "utc"
        }
        sentence = sentence.lower().replace("forty", "fourty")
        for key, value in conv_dict.items():
            sentence = sentence.replace(key, value)
        return sentence
