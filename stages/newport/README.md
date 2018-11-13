# modprobe ti_usb_3410_5052
# /bin/sh -c ‘echo 104d 3001 > /sys/bus/usb-serial/drivers/ti_usb_3410_5052_1/new_id
# /bin/sh -c ‘echo 104d 3001 > /sys/bus/usb-serial/drivers/ti_usb_3410_5052_2/new_id

# modprobe usbserial vendor=0x104d product=0x3001


#To automate this, add the following to /etc/udev/rules.d/99-ftdi.rules:

#ACTION==”add”, ATTRS{idVendor}==”104d″, ATTRS{idProduct}==”3001″, RUN+=”/sbin/modprobe ftdi_sio” RUN+=”/bin/sh -c ‘echo 104d 3001 > /sys/bus/usb-serial/drivers/ftdi_sio/new_id’”


# modprobe ti_usb_3410_5052
# /bin/sh -c ‘echo 104d 3001 > /sys/bus/usb-serial/drivers/ti_usb_3410_5052_1/new_id
# /bin/sh -c ‘echo 104d 3001 > /sys/bus/usb-serial/drivers/ti_usb_3410_5052_2/new_id

