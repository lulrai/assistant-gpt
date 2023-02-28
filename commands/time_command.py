""" Command to read the current time. """
from datetime import datetime, timezone, timedelta
import regex
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
            timezone_regex = regex.compile(r"(UTC).*([+])+.*([0-2]*[0-9]).*(00|15|30|45)*", regex.IGNORECASE)
            timezone_val = args["timezone:timezone"][0]["body"]
            timezone_str = self.__num_words_to_num(timezone_val)
            timezone_match = timezone_regex.match(timezone_str)
            if timezone_match:
                timezone_sign = timezone_match.group(2)
                timezone_hour = int(timezone_match.group(3))
                timezone_minute = int(timezone_match.group(4)) if timezone_match.group(4) else 0
                if timezone_sign == "+":
                    timezone_offset = timedelta(hours=timezone_hour, minutes=timezone_minute)
                else:
                    timezone_offset = -timedelta(hours=timezone_hour, minutes=timezone_minute)

                timezone_val = timezone(timezone_offset)
                current_time = datetime.now(timezone_val)
                # convert the UTC datetime to the specified timezone
                time_at_offset = current_time.astimezone(timezone_val)
                day_ordinal = self.ordinal(time_at_offset.day)
                greeting = self.greeting(time_at_offset.hour)
                time_string = current_time.strftime(f'It is %-I:%M %p on %A of %B {day_ordinal}, %Y in %Z')
            else:
                time_string = "Sorry, I don't understand the timezone you specified."
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
