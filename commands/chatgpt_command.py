""" Command to ask Chat GPT a question. """
import os
from revChatGPT.V1 import Chatbot
from commands.command import Command


class ChatGPTCommand(Command):
    """ This class is the command to ask Chat GPT a question. """
    def __init__(self) -> None:
        self.command_aliases = ['chatgpt', 'gpt', 'ask']
        super().__init__(command_aliases = self.command_aliases, intent = "wit_chatgpt", requires_args = True)
        self.chatbot = Chatbot(config={
            "access_token": os.environ["CHAT_GPT_AUTH"],
        })

    def execute_string_command(self, args) -> str:
        for alias in self.command_aliases:
            if alias in args:
                args.remove(alias)
        question = " ".join(args)
        answer = ""
        for data in self.chatbot.ask(question):
            answer = data["message"]
        return answer

    def execute_intent_command(self, args) -> str:
        gpt_question_entity = args["gpt_question:gpt_question"]
        question_text = gpt_question_entity[0]["body"]
        for data in self.chatbot.ask(question_text):
            answer = data["message"]
        return answer