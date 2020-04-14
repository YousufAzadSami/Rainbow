def inputTestWtihInput():
    print("1")
    yourvar = input('Choose a number: ')
    print('you entered: ' + yourvar)
    print("2")

import readchar

# print("Reading a char:")
# print(repr(readchar.readchar()))
print("Reading a key:")
char = repr(readchar.readkey())
print (char)
action = 100

if (char == "'a'"):
    action = -1
elif (char == "'s'"):
    action = 0
else:
    action = +1

print(action)

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()