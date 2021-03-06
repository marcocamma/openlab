
import re
import sys

_time2value = dict( fs = 1e-15, ps = 1e-12, ns = 1e-9, us = 1e-6, ms = 1e-3, s = 1)

_time_regex = re.compile("(-?\d+\.?\d*)((?:s|fs|ms|ns|ps|us)?)")

def str_to_time(delay) :
    if isinstance(delay,(float,int)): return delay
    match = _time_regex.search(delay)
    if match:
        n,t = float(match.group(1)),match.group(2)
        value = _time2value.get(t,1)
        return n*value
    else:
        return None

def time_to_str(delay,fmt="%+.0f"):
    try:
        a_delay = abs(delay)
    except TypeError:
        return None
    if a_delay >= 1:
        ret = fmt % delay + "s"
    elif 1e-3 <= a_delay < 1: 
        ret = fmt % (delay*1e3) + "ms"
    elif 1e-6 <= a_delay < 1e-3: 
        ret = fmt % (delay*1e6) + "us"
    elif 1e-9 <= a_delay < 1e-6: 
        ret = fmt % (delay*1e9) + "ns"
    elif 1e-12 <= a_delay < 1e-9: 
        ret = fmt % (delay*1e12) + "ps"
    elif 1e-15 <= a_delay < 1e-12: 
        ret = fmt % (delay*1e12) + "fs"
    elif 1e-18 <= a_delay < 1e-15: 
        ret = fmt % (delay*1e12) + "as"
    else:
        ret = str(delay) +"s"
    return ret

def terminal_size():
    """returns a tuple with (nrows,ncols) of current terminal"""
    import termios, fcntl, struct, sys
    s = struct.pack("HHHH", 0, 0, 0, 0)
    fd_stdout = sys.stdout.fileno()
    x = fcntl.ioctl(fd_stdout, termios.TIOCGWINSZ, s)
    (rows,cols,xpixel,ypixel)=  struct.unpack("HHHH", x)
    return (rows,cols)


def notice(string):
    """prints a string starting from the beginning of the line and spanning the
    entire terminal columns, useful for updating values ...
    notice("this is the first value 1.2"); sleep(1); notice("this is the second one 1.4 that will mask the first...")
    """
    (nrows,ncols)=terminal_size()
    format = "\r%%-%ds" % ncols
    sys.stdout.write( format % string)
    sys.stdout.flush()

