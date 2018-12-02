import zmq
import argparse
import datetime
import time
import sys
from . import temper

reader=temper.Temper()

def now():
    return datetime.datetime.now()

def read(n_average=5,sleep=2):
    temp = 0
    hum = 0
    n = 0
    for i in range(n_average):
        try:
            data = reader.read()[0]
            temp += data["internal temperature"]
            hum  += data["internal humidity"]
            n    += 1
        except:
            pass
        time.sleep(sleep)
    temp = temp/n
    hum = hum/n
    return dict(temperature=temp,humidity=hum)

def read_and_publish(sensor_name = "cabanne", n_average=5,port=1234):
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    addr = "tcp://*:%s"%port
    sock.bind(addr)
    while True:
        try:
            data = read(n_average=n_average)
            tosend = dict()
            for info,value in data.items():
                tosend[sensor_name+"/"+info] = value
            tosend["time"] = now()
            print(tosend)
            sock.send_pyobj(tosend)
        except KeyboardInterrupt:
            print("Detected ctrl+c, exiting")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process argv")

    parser.add_argument("--name", default="sensor", help="publish data as sensor/{temperature,humidity}")
    parser.add_argument("--n",type=int,default=5,help="number of local averages before sending out the data")

    args = parser.parse_args()

    reader = read_and_publish(
            sensor_name=args.name,
            n_average=args.n)
