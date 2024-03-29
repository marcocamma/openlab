"""
written to be used from scope modules, useful for something else ?
"""
import socket
import os
import time
DEBUG=False
DEBUG=True

PORT = 4000

class SocketConnection:

    def __init__(self, host="129.20.76.100", port=PORT,waittime=0.001,encoding='utf8'):
        self.host = host
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        self.s.settimeout(5)
        self.waittime = waittime
        self.encoding=encoding

        self.f = self.s.makefile("rb")

    def write(self,msg):
        if DEBUG: print("Writing",msg)
        if not msg.endswith("\n"): msg += "\n"
        self.s.send(msg.encode("ascii"))
        time.sleep(self.waittime)

    def read_raw(self):
        """
        Return a message from the scope.
        """
        reply = b""
        blocksize = 1<<16
        if DEBUG: print("Reading",end=" ")
        t0 = time.time()
        while not reply.endswith("\n".encode("ascii")):
            try:
                #reply += self.s.recvmsg(blocksize)[0]
                reply += self.s.recv(blocksize)
            except socket.timeout:
                print("Failed to read until end of message")
        if DEBUG: print("... read %d in %.3f s"%(len(reply),time.time()-t0))
        if DEBUG: print("... last bytes",reply[-10:])
        return reply

    def read_nbytes(self,nbytes=100):
        """
        Return a message from the scope.
        """
        reply = b""
        while len(reply) < nbytes:
            reply += self.s.recv(nbytes-len(reply))
        return reply

    def read_nbytes_preallocate(self,nbytes=100):
        """
        Return a message from the scope.
        """
        reply = bytearray(nbytes)
        last_idx = 0
        while last_idx < nbytes:
            received = self.s.recv(nbytes-last_idx)
            reply[last_idx:last_idx+len(received)] = received
            last_idx += len(received)
        return bytes(reply)


    def read(self,encoding='auto'):
        """
        Return a message from the scope.
        """
        if encoding == "auto" : encoding = self.encoding
        reply = self.read_raw()
        reply = reply.decode(encoding)
        reply = reply.strip()
        return reply

    def ask(self,msg,encoding='auto'):
        if encoding == "auto" : encoding = self.encoding
        if not msg.endswith("?"): msg += "?"
        self.write(msg)
        return self.read(encoding=encoding)

    def ask_raw(self,msg):
        if not msg.endswith("?"): msg += "?"
        self.write(msg)
        return self.read_raw()



class USBTMC:

    def __init__(self, port="/dev/usbtmc1",terminator="\n",encoding="utf8"):

        self._file = os.open(port,os.O_RDWR)
        self._terminator = terminator
        self.encoding=encoding

    def write(self,msg):
        if not msg.endswith(self._terminator): msg += self._terminator
        os.write(self._file,msg.encode(self.encoding))

    def read(self,encoding='auto'):
        """
        Return a message from the scope.
        """
        if encoding == "auto" : encoding = self.encoding
        if not msg.endswith("?"): msg += "?"
        raw = self.read_raw()
        reply = raw.decode(encoding)
        reply = reply.strip()
        return reply

    def read_raw(self):
        reply = b''
        terminator = self._terminator.encode('ascii')
        while not reply.endswith(terminator):
            reply += os.read(self._file,1<<30)
        return reply

    def read_nbytes(self,nbytes=100):
        """
        Return a message from the scope.
        """
        reply = b""
        while len(reply) < nbytes:
            reply += os.read(self._file,nbytes-len(reply))
        return reply


    def ask_raw(self,msg):
        if not msg.endswith("?"): msg += "?"
        self.write(msg)
        return self.read_raw()

    def ask(self,msg,encoding='auto'):
        if encoding == "auto" : encoding = self.encoding
        if not msg.endswith("?"): msg += "?"
        if not msg.endswith("?"): msg += "?"
        self.write(msg)
        return self.read(encoding=encoding)





