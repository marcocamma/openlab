import socket
import pathlib
import collections
import enum
import time
import shutil

from ..utils import yaml_storage

STORAGE = yaml_storage.Storage(filename="./offsets/bme_data.yaml", autosave=False)

DEFAULT_IP = "129.20.76.14"

VERBOSE = False
VERBOSE = True



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
        STORAGE[self.save_field]=self.read_delay()

    def read(self):
        value = STORAGE[self.save_field]
        self.move_delay(value)

    def __repr__(self):
        return self.info(verbose=True)

class BME_CARD:
    def __init__(self,connection,card_num=1,n_channels=6):
        """ times are in us """
        self.channels = "ABCDEF"[:n_channels]
        self.connection=connection
        self.card_num = card_num
        for channel in self.channels:
            setattr(self,channel,Trigger(self,channel=channel))

    def channel_bits(self,channel="A",value=None,auto_apply=True):

        if channel not in self.channels:
            raise ValueError("channel can only be one of the characters %s"%\
                    self.channels)

        if value is None: # means read
            cmd = "DLAC? %d,%s" % (self.card_num,channel)
            ret = int(self.connection.query(cmd))
            ret = int(self.connection.query(cmd)) # ask twice, first time gets old values sometime ..
            return channel_bits.get_bits(ret)
        else:
            value = channel_bits.get_int(value)
            cmd = "DLAC %d,%s,%d" % (self.card_num,channel,value)
            self.connection.send(cmd)
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
            self.connection.send(cmd_write)
            if auto_apply:
                self.connection.apply()
                ret = self.connection.query(cmd)
            else:
                ret = value
        else:
            ret = self.connection.query(cmd)

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
        self.connection.apply()

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


class BME_BOX:

    def __init__(self,ip=DEFAULT_IP,port=8002,timeout=0.3):
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

        self.dg1 = BME_CARD(self,card_num=1)
        self.dg2 = BME_CARD(self,card_num=2)
        self.dg3 = BME_CARD(self,card_num=3)
        self.dg4 = BME_CARD(self,card_num=4)

    def send(self,string):
        string = string + "\r"
        if VERBOSE: print("send %s"%string)
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
                if VERBOSE: print("got timeout, retrying")
        if VERBOSE: print("got",answer.strip())
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

class AMPLIFIER_TIMING:
    def __init__(self,amplifier_card=None,
            amplifier_name="elite",pc_relative_delay=None,burst_card=None,
            do_autosetup=False):
        """
        pc_relative_delay is with respect to pump ... time in sec
        amplifier_name used for storing offset
        """
        self.amplifier_name=amplifier_name
        self.amplifier_card=amplifier_card
        self.pump_trigger = amplifier_card.A
        self.pc_trigger = amplifier_card.B
        self.chopper_trigger = amplifier_card.C
        self.burst_card = burst_card
        if burst_card is not None:
            self.burst_gate = self.amplifier_card.F
            self.shutter_trigger = burst_card.A
            self.shutter_gate = burst_card.B
            self.detector_trigger = burst_card.C
            self.scope_trigger = burst_card.D

        if pc_relative_delay is None:
            pc_relative_delay = self.storage("pc_delay",default=0)

        if do_autosetup:
            self.setup_cards()

        self.pc_relative_delay = pc_relative_delay
        self.set_pc_relative_delay(pc_relative_delay)


        self._set_chopper_duty_cycle()
        self.shutter_disable()


    def setup_cards(self):
        self.pump_trigger.name="Pump"
        self.pump_trigger.channel_bits(value='master_primary')
        self.pc_trigger.name = "Pockell"
        self.pc_trigger.channel_bits(value='master_primary')
        self.chopper_trigger.name="Chopper"
        self.chopper_trigger.channel_bits(value='master_primary')
        self.chopper_divider(2)
        self.amplifier_card.apply()
        name = self.amplifier_name + "_on"
        self.amplifier_card.cmd('card_name',value=name)
        if self.burst_card is not None:
            name = self.amplifier_name + "_burst"
            self.burst_card.cmd('card_name',value=name)
            self.burst_card.cmd('gate_termination',value='highZ')
            self.burst_gate.name = "BurstGate"
            self.burst_gate.termination = "highZ"
            self.burst_gate.delay="A-0.1"
            self.burst_gate.width="P*0.99"
            self.burst_gate.divider="CM"
            time.sleep(3); # do not know why it is necessary
            self.burst_gate.channel_bits(value='master_primary')
            burst = self.burst_card
            self.shutter_disable()
            self.burst_card.connection.send("LINK %d,AB,AND"%self.burst_card.card_num)
            self.shutter_trigger.name="Shutter"
            self.shutter_trigger.channel_bits(value='local_primary')
            self.shutter_gate.name = "ShutterGate"
            self.shutter_gate.channel_bits(value='local_primary')
            self.detector_trigger.name = "Detector"
            self.detector_trigger.channel_bits(value='local_primary')
            self.scope_trigger.name = "Scope"
            self.scope_trigger.channel_bits(value='local_primary')
            self.burst_card.apply()
#            burst.cmd('gate_source',value='external_raising')
            burst.cmd('gate_ignore',value='true')
            burst.cmd('gate_level',value=1.2)
            self.burst_card.apply()
#            burst.A.cmd('

    def storage(self,what,value=None,default=0):
        what = self.amplifier_name + "/" + what
        if value is None:
            return STORAGE.get(what,default)
        else:
            STORAGE[what] = value

    def save(self,fname=None,backup=True):
        current_delay = self.read_amplifier_delay()
        lname = self.amplifier_name

        # temporaly move to zero delay to read offsets
        self.move_amplifier_delay(0)
        self.storage('pump_delay_offset',
            value=self.read_amplifier_delay(as_dial=True) )
        self.storage('pc_delay',value=self.pc_trigger.read_delay())
        STORAGE.save(fname=fname,backup=backup)

        # move back to previous values
        self.move_amplifier_delay(current_delay)

    def read(self,fname=None):
        if fname is None: fname = STORAGE._filename
        temp = yaml_STORAGE.Storage(fname)
        STORAGE['%s/pump_delay_offset'%lname]=temp['%s/pump_delay_offset'%lname]
        STORAGE['%s/pc_delay'%lname]=temp['%s/pc_delay'%lname]
        self.pc_trigger.move_delay(temp['%s/pc_delay'%lname])
        self.move_amplifier_delay(0)


    def set_pc_relative_delay(self,pc_relative_delay=None):
        if pc_relative_delay is None: pc_relative_delay = self.pc_relative_delay
        if isinstance(pc_relative_delay,(float,int)):
            delay = self.pump_trigger.channel +"+%.6f"%(pc_relative_delay*1e6)
        else:
            delay = pc_relative_delay
        self.pc_trigger.move_delay(delay)

    def move_amplifier_delay(self,value,as_dial=False,auto_apply=True):
        if not as_dial:
            offset = STORAGE['%s/pump_delay_offset'%self.amplifier_name]
            value = value+offset
        self.pump_trigger.move_delay(value*1e6)
        if auto_apply: self.amplifier_card.apply()

    def set_amplifier_delay(self,value):
        current_dial = self.read_amplifier_delay(as_dial=True)
        newoffset = current_dial-value
        STORAGE['%s/pump_delay_offset'%self.amplifier_name] = newoffset

    def read_amplifier_delay(self,as_dial=False):
        value = float(self.pump_trigger.read_delay())*1e-6
        if not as_dial:
            offset = STORAGE['%s/pump_delay_offset'%self.amplifier_name]
            value = value - offset
        return value

    def _set_chopper_duty_cycle(self):
        amplifier_period = float(self.amplifier_card.cmd('clock_inhibit'))*int(self.pc_trigger.divider)
        chopper_width = amplifier_period*float(self.chopper_trigger.divider)/2
        self.chopper_trigger.width = chopper_width


    def chopper_divider(self,n=2):
        self.chopper_trigger.divider = n
        # set duty cycle to 50%
        self._set_chopper_duty_cycle()

    def shutter_disable(self):
        # actually never fires because not used, completely disabling would 
        # cancel delay and width information
        self.shutter_trigger.channel_bits(value='local_force')

    def shutter_enable(self,auto_apply=True):
        #if n<100:
        #    raise ValueError("too high of a frequency for the shutter")
        #else:
        self.shutter_trigger.channel_bits(value='local_primary',auto_apply=auto_apply)



    def fire(self,n=None,nbefore=100):
        if self.burst_card is None: return
        if n is None:
            self.burst_card.cmd('stop_at_n_pulses',value='false')
        else:
            self.shutter_enable(auto_apply=True)
            delay = self.shutter_trigger.read_delay()
            if delay.find("+") > 0: delay = delay.split("+")[1] # takes only 'fine delay'
            delay = delay.strip()
            self.shutter_gate.move_delay(delay)
            self.shutter_gate.width = self.shutter_trigger.width
            chopper_divider = self.chopper_trigger.divider
            delay = "%d*%s*P%d + %s" % (nbefore,chopper_divider,self.amplifier_card.card_num,delay)
            self.shutter_trigger.move_delay(delay)
            self.burst_card.cmd('stop_at_n_pulses',value='true',auto_apply=False)
            self.burst_card.cmd('n_pulses',value=int(n+2*nbefore),auto_apply=False)
            self.burst_card.apply()

    def __repr__(self):
        dial = self.read_amplifier_delay(as_dial=True)
        user = self.read_amplifier_delay(as_dial=False)
        s = 'Amplifier "%s", delay (user,dial): %.12f %.12f\n'%(self.amplifier_name,
                user,dial)
        s += "Pump trigger    : %s\n"%str(self.pump_trigger)
        s += "PC trigger      : %s\n"%str(self.pc_trigger)
        if self.chopper_trigger is not None:
            s += "Chopper trigger : %s\n" % str(self.chopper_trigger)
        if self.burst_card is not None:
            s += "Burst gate      : %s\n" % str(self.burst_gate)
            s += "Gated Output\n"
            s += " → Shutter      : %s\n" % str(self.shutter_trigger)
            s += " → Detector     : %s\n" % str(self.detector_trigger)
            s += " → Scope        : %s\n" % str(self.scope_trigger)
#
        return s.strip()
        

box = BME_BOX()
if box.connection is not None:
    amplifier1 = AMPLIFIER_TIMING(amplifier_card=box.dg1,amplifier_name="elite",
        burst_card=box.dg2)
    amplifier2 = AMPLIFIER_TIMING(amplifier_card=box.dg3,amplifier_name="usp",
        burst_card=box.dg4)







