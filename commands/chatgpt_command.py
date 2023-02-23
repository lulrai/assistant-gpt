""" Command to ask Chat GPT a question. """
import os
from revChatGPT.V1 import Chatbot
from commands.command import Command


class ChatGPTCommand(Command):
    """ This class is the command to ask Chat GPT a question. """
    def __init__(self) -> None:
        self.command_aliases = ['chatgpt', 'gpt', 'ask']
        super().__init__(command_aliases = self.command_aliases, requires_args = True)
        self.chatbot = Chatbot(config={
            "access_token": os.environ["CHAT_GPT_AUTH"],
        })

    def execute_command(self, args) -> str:
        for alias in self.command_aliases:
            if alias in args:
                args.remove(alias)
        question = " ".join(args)
        answer = ""
        for data in self.chatbot.ask(question):
            answer = data["message"]
        return answer
