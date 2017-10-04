#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client

import argparse
import time
import cv2
import json
import os

run_num = 1
folder = os.path.join("D:", "GTAVDataset", "run%i" % run_num)
json_storage = [""]
if not os.path.isdir(folder):
    os.mkdir(folder)
img_prefix = os.path.join(folder, "image%i.png")
dat_prefix = os.path.join(folder, "meta%i.json")
top_offset = 30
bot_offset = 30

size = (227 + top_offset + bot_offset, 227)

class Model:
    def run(self,frame):
        return [0.5, 0.0, 0.0] # throttle, brake, steering

# Controls the DeepGTAV vehicle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port.
    # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
    # We don't want to save a dataset in this case
    client = Client(ip=args.host, port=args.port)

    # We set the scenario to be in manual driving, and everything else random (time, weather and location).
    # See deepgtav/messages.py to see what options are supported
    scenario = Scenario(drivingMode=-1, weather="CLEAR", vehicle="blista") #manual driving

    dataset = Dataset(
        rate=1,
        frame=size,
        vehicles=True,
        peds=False,
        trafficSigns=False,
        direction=None, # What is this param?
        reward=None,
        throttle=False,
        brake=False,
        steering=False,
        yawRate=False,
        drivingMode=True,
        location=True,
        heading=True,
        time=False,
        )

    # Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
    client.sendMessage(Start(scenario=scenario, dataset=dataset))

    # Dummy agent
    model = Model()

    # Start listening for messages coming from DeepGTAV. We do it for 80 hours
    stoptime = time.time() + 80*3600
    cv2.namedWindow("frame")
    count = 0
    while time.time() < stoptime:
        try:
            # We receive a message as a Python dictionary
            message = client.recvMessage()

            dat = {
                "heading": message["heading"],
                "location": message["location"]
                }
            print(dat)

            # The frame is a numpy array that can we pass through a CNN for example
            image = frame2numpy(message['frame'], size)
            image = image[top_offset:-bot_offset,:]

            # SAVING
            #cv2.imshow("frame", image)
            #cv2.waitKey(1)
            cv2.imwrite(img_prefix % count, image)
            with open(dat_prefix % count, "w+") as f:
                json.dump(dat, f)

            # CONTROLS
            commands = model.run(image)
            # We send the commands predicted by the agent back to DeepGTAV to control the vehicle
            client.sendMessage(Commands(commands[0], commands[1], commands[2]))
            count += 1
        except KeyboardInterrupt:
            break

    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()
