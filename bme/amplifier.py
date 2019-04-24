import socket
import pathlib
import collections
import enum
import time
import shutil


class AmplifierTiming:
    def __init__(self,amplifier_card=None,
            amplifier_name="elite",pc_relative_delay=None,burst_card=None,
            do_autosetup=False):
        """
        pc_relative_delay is with respect to pump ... time in sec
        amplifier_name used for storing offset
        """
        self._box_storage = amplifier_card.box.storage
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
        # self.shutter_disable() # commented because we are using it as signal for second chopper


    def setup_cards(self):
        self.pump_trigger.name="Pump"
        self.pump_trigger.channel_bits(value='master_primary')
        self.pump_trigger.width=10
        self.pc_trigger.name = "Pockell"
        self.pc_trigger.channel_bits(value='master_primary')
        self.pc_trigger.width=10
        self.chopper_trigger.name="Chopper"
        self.chopper_trigger.channel_bits(value='master_primary')
        self.amplifier_card.cmd('clock_inhibit',value=1000)
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
            self.burst_card.box.send("LINK %d,AB,AND"%self.burst_card.card_num)
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
            return self._box_storage.get(what,default)
        else:
            self._box_storage[what] = value

    def save(self,fname=None,backup=True):
        current_delay = self.read_amplifier_delay()
        lname = self.amplifier_name

        # temporaly move to zero delay to read offsets
        self.move_amplifier_delay(0)
        self.storage('pump_delay_offset',
            value=self.read_amplifier_delay(as_dial=True) )
        self.storage('pc_delay',value=self.pc_trigger.read_delay())
        self._box_storage.save(fname=fname,backup=backup)

        # move back to previous values
        self.move_amplifier_delay(current_delay)

    def read(self,fname=None):
        if fname is None: fname = self._box_storage._filename
        temp = yaml_storage.Storage(fname)
        self.storage('pump_delay_offset',value=temp['%s/pump_delay_offset'%lname])
        self.storage('pc_delay',value=temp['%s/pc_delay'%lname])
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
            offset = self._box_storage['%s/pump_delay_offset'%self.amplifier_name]
            value = value+offset
        self.pump_trigger.move_delay(value*1e6)
        if auto_apply: self.amplifier_card.apply()

    def set_amplifier_delay(self,value):
        current_dial = self.read_amplifier_delay(as_dial=True)
        newoffset = current_dial-value
        self.storage('pump_delay_offset',value=newoffset)
        self.save()

    def read_amplifier_delay(self,as_dial=False):
        value = float(self.pump_trigger.read_delay())*1e-6
        if not as_dial:
            offset = self.storage('pump_delay_offset')
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
