import socket
import pathlib
import collections
import enum
import time

from ..utils import yaml_storage


##########################
# SET/GET VALUES MAPPING #
##########################

class _true_false(enum.Enum):
    false = 0
    true = 1

class _trigger_source(enum.Enum):
    internal = 0
    external_raising = 1 # rising edge
    external_falling = 2 # falling_edge

class _termination(enum.Enum):
    lowZ = 0
    highZ = 1

class _Channel_Bits:
    mapping = dict(
        local_primary    = 0x1,
        local_secondary  = 0x2,
        local_force      = 0x4,
        resync           = 0x8,
        master_primary   = 0x10,
        master_secondary = 0x20,
        master_force     = 0x40,
        system_clock     = 0x80,
        delay_clock      = 0x100,
        inhibit_local    = 0x200,
        start_local      = 0x400,
        start_bus        = 0x800,
        step_back_local  = 0x1000,
        step_back_bus    = 0x2000,
        run_circle       = 0x4000,
        sync_reload      = 0x8000,
    )

    def get_int(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            args = args[0].split()
        tot = 0
        for arg in args:
            if arg in self.mapping:
                tot += self.mapping[arg]
            else:
                print("Attribute %s does not exists")
        return tot

    def get_bits(self, value):
        mapping = self.mapping
        ret = collections.OrderedDict()
        keys = sorted(mapping, key=mapping.get, reverse=True)
        for key in keys:
            if value // mapping[key] > 0:
                ret[key] = True
                value = value - mapping[key]
            else:
                ret[key] = False
        return ret


channel_bits = _Channel_Bits()



####################################
# COMMAND NAME -> MNEMONIC MAPPING #
####################################

_cmd_channel = dict(
    delay   = 'DLAY',
    go      = 'DLAC',
    divider = 'DLAM',
    width   = 'DLAW',
    name    = 'CNAM',
    termination = 'DLTM'
)

_cmd_card = dict(
    burst       = 'BRST',
    card_name   = 'DNAM',
    gate_source = 'GATE',
    gate_ignore = 'GIGN',
    gate_level  = 'GLVL',
    gate_termination    = 'TRMG',
    clock_divider       = 'CEDV',
    clock_inhibit       = 'INHB',
    trigger_source      = 'TSRC',
    trigger_level       = 'TLVL',
    trigger_termination = 'TRMT',
    n_pulses    = 'PRES',
    stop_at_n_pulses    = 'STPR'
)

_converter = dict(
    trigger_source      = _trigger_source,
    trigger_termination = _termination,
    termination         = _termination,
    gate_source         = _trigger_source,
    gate_ignore         = _true_false,
    gate_termination    = _termination,
    stop_at_n_pulses    = _true_false
)

class Trigger:
    def __init__(self, bme_card, channel="A"):
        self.bme_card = bme_card
        self.channel = channel
        self.save_field = "/".join( (str(self.bme_card.card_num),channel,"delay") )

    def cmd(self, what, auto_apply=True, **kwargs):
        """ 
        This function checks if the value has changed before chaning value.
        This is done because reading takes 1ms, setting 200ms ...
        """
        if 'value' in kwargs:
            present_value = self.bme_card.cmd(what,channel=self.channel)
            if present_value == kwargs['value']:
                return present_value
        return self.bme_card.cmd(what,channel=self.channel,
                auto_apply=auto_apply,**kwargs)

    def __setattr__(self,what,value):
        if what in _cmd_channel:
            self.cmd(what,value=value)
        else:
            return super().__setattr__(what, value)

    def __getattr__(self,what):
        if what in _cmd_channel:
            return self.cmd(what)
        else:
            return super().__getattr__(what, value)

    def channel_bits(self, value=None, auto_apply=True):
        return self.bme_card.channel_bits(channel=self.channel, value=value,
                auto_apply=auto_apply)

    def read_delay(self):
        value = self.cmd("delay")
        return value

    def move_delay(self,value):
        value = self.cmd("delay", value=value)

    def info(self,verbose=True):
        s = 'card %s, '%self.bme_card.card_num
        s += 'output %s, '%self.channel
        if verbose:
            s += 'name %-10s, '%self.cmd('name')
            s += 'termination %5s, ' % self.cmd('termination')
        s += 'delay %sus' % self.read_delay()
        if verbose:
            divider = self.cmd('divider', as_type=int)
            if not isinstance(divider,int) or divider>1:
                s += ', divider %s' %divider
            s += ', width %sus' % self.cmd('width')
        return s

    def apply(self): self.bme_card.apply()

    def save(self):
        self.bme_card.storage[self.save_field]=self.read_delay()

    def read(self):
        value = self.bme_card.storage[self.save_field]
        self.move_delay(value)

    def __repr__(self):
        return self.info(verbose=True)

class BmeCard:
    def __init__(self,box,card_num=1,n_channels=6):
        """ times are in us """
        self.channels = "ABCDEF"[:n_channels]
        self.box = box
        self.card_num = card_num
        for channel in self.channels:
            setattr(self,channel,Trigger(self,channel=channel))

    def channel_bits(self,channel="A",value=None,auto_apply=True):

        if channel not in self.channels:
            raise ValueError("channel can only be one of the characters %s"%\
                    self.channels)

        if value is None: # means read
            cmd = "DLAC? %d,%s" % (self.card_num,channel)
            ret = int(self.box.query(cmd))
            ret = int(self.box.query(cmd)) # ask twice, first time gets old values sometime ..
            return channel_bits.get_bits(ret)
        else:
            value = channel_bits.get_int(value)
            cmd = "DLAC %d,%s,%d" % (self.card_num,channel,value)
            self.box.send(cmd)
            if auto_apply: self.apply()

    def cmd(self,what,auto_apply=True,**kwargs):
        # special care for enable/starts bits
        if what == 'channel_bits':
            return self.channel_bits(channel=kwargs['channel'],
                    value=kwargs.get('value'))

        channel = kwargs.get('channel',None)
        value = kwargs.get('value',None)

        if channel is not None:
            if channel not in self.channels:
                raise ValueError("channel can only be one of the characters %s"%\
                        self.channels)
            cmd = _cmd_channel[what]
            cmd = "%s? %d,%s" % (cmd,self.card_num,channel)
        else:
            cmd = _cmd_card[what]
            cmd = "%s? %d" % (cmd,self.card_num)

        if value is not None:

            # convert if property has an enum converter
            if (what in _converter) and \
                isinstance(_converter[what],enum.EnumMeta):
                value = _converter[what][value].value

            cmd_write = cmd.replace("?"," ")
            cmd_write += ",%s" % value
            self.box.send(cmd_write)
            if auto_apply:
                self.box.apply()
                ret = self.box.query(cmd)
            else:
                ret = value
        else:
            ret = self.box.query(cmd)

        if what in _converter and \
            isinstance(_converter[what],enum.EnumMeta):
            ret = _converter[what](int(ret)).name

        if 'as_type' in kwargs:
            try:
                ret = kwargs['as_type'](ret)
            except ValueError:
                pass

        return ret


    def apply(self):
        self.box.apply()

    def default_widths(self):
        for channel in self.channels:
            self.cmd('width',channel=channel,auto_apply=False,value=5e-6)
        self.apply()

    def set_defaults(self):
        """ Put card in a default state (1kHz trigger, master_primary start, 5us delay and width """

        self.cmd('trigger_level',value=0.05)
        self.cmd('clock_inhibit',value=1000); # 1000us -> 1KHz
        self.cmd("trigger_source",value="internal")

        # enable channels first
        for channel in self.channels:
            self.cmd('channel_bits',channel=channel,auto_apply=False,value="master_primary")
        self.apply()

        for channel in self.channels:
            self.cmd('delay',channel=channel,auto_apply=False,value=5)
            self.cmd('width',channel=channel,auto_apply=False,value=5)
        self.apply()

    def __repr__(self):
        name = 'Card: %s\n' % self.cmd('card_name')
        triggers = "\n".join( [str(getattr(self,channel)) for channel in self.channels])
        return name + triggers


class BmeBox:

    def __init__(self,ip=None,port=8002,timeout=0.3,
            storage=None,verbose=True):
        """
        storage is an instance of yaml_storage, if None the default is used
        """
        if storage is None:
            storage = yaml_storage.Storage(filename="./offsets/bme_data.yaml", autosave=False)
        self.storage = storage
        self.verbose = verbose
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((ip, port))
            self.connection.settimeout(timeout)
        except OSError:
            print("Cannot connect to the computer %s"%ip)
            self.connection = None
            return



        # test connection #
        idn = self.query('*IDN?')
        if idn == '':
            msg = 'Cannot connect to the box, either is not on the'
            msg += 'network or there is already a connected client'
            raise ConnectionError(msg)

        self.dg1 = BmeCard(self,card_num=1)
        self.dg2 = BmeCard(self,card_num=2)
        self.dg3 = BmeCard(self,card_num=3)
        self.dg4 = BmeCard(self,card_num=4)

    def send(self,string):
        string = string + "\r"
        if self.verbose: print("bmebox send %s"%string)
        byte_string = string.encode('ascii')
        self.connection.send(byte_string)

    def get(self,bufsize=1024,as_type=None,n_attempts=5):
        answer=""
        for i in range(n_attempts):
            try:
                answer = self.connection.recv(bufsize).decode('ascii')
                break
            except socket.timeout:
                time.sleep(0.3)
                if self.verbose: print("bmebox got timeout, retrying")
        if self.verbose: print("bmebox got",answer.strip())
        if as_type is not None: answer = as_type(answer)
        return answer

    def query(self,string,bufsize=1024,as_type=None,n_attempts=5):
        self.send(string)
        answer = self.get(bufsize=bufsize,as_type=as_type,n_attempts=n_attempts)
        return answer.strip()

    def apply(self):
        self.send("APLY")

    def __del__(self):
        self.connection.close()


