import sys
import select
import tty
import termios
import time

arrow_up   = '\x1b[A'
arrow_UP   = '\x1b[1'
arrow_down = '\x1b[B'
arrow_right = '\x1b[C'
arrow_left = '\x1b[D'

def isData():
  return select.select([sys.stdin], [], [], 1) == ([sys.stdin], [], [])

def getc():
  """ Non blocking get character """
  old_settings = termios.tcgetattr(sys.stdin)
  c=None
  try:
    tty.setcbreak(sys.stdin.fileno())
    if isData():
      c = sys.stdin.read(1)
  finally:
      termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
  return c


def wait_input():
  """ wait for a character and returns it """
  old_settings = termios.tcgetattr(sys.stdin)
  c=None
  try:
    tty.setcbreak(sys.stdin.fileno())
    while (not isData()):
      time.sleep(0.001)
    c = sys.stdin.read(1)
    if ( (c == '\x1b') ):
      c = c + sys.stdin.read(2)
    return c
  finally:
      termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
  return c

def wait_char(char='q'):
  """ wait for a specific character, default is q """
  c = None
  while (c != char):
    c=wait_input()
#    print "got |%s| but waiting for |%s|" % (c,char)

class KeyPress:
  def __init__(self,esc_key="q",debug=False):
    self.esc_key=esc_key
    self.debug = debug
    self.last_key = None

  def waitkey(self):
    a = wait_input()
    self.last_key=a
    return a

  def isu(self):
    return (self.last_key==arrow_up)
  def isU(self):
    return (self.last_key==arrow_UP)
  def isd(self):
    return (self.last_key==arrow_down)
  def isl(self):
    return (self.last_key==arrow_left)
  def isr(self):
    return (self.last_key==arrow_right)
  def isq(self):
    return (self.last_key==self.esc_key)
  def iskey(self,what):
    return (self.last_key==what)
#wait_char('\x37')
#s=wait_input()
#print (s==arrow_up)

if (__name__ == "__main__"):
  while(1):
    print("press a key (q to exit)")
    a=wait_input()
    print("|%s|" % a)
    if (a=="q"): break
