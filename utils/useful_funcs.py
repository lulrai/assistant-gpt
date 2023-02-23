""" This module is used to keep useful functions. """
import os
import sys
import contextlib

@contextlib.contextmanager
def silence():
    """ This function is used to hide the prints from the console. """
    sys.stdout, old = open(os.devnull, 'w', encoding="utf-8"), sys.stdout
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
