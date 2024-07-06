# simple python script to measure latency between "here" and some other IP address,
# using ICMP echo requests (ping), then log the result using a Google Sheet endpoint.

# SEE AWS LAMBDA FOR THE ACTUAL VERSION OF THIS

import os
import sys
import time
import subprocess
import requests
import json
import datetime

# Google Sheet endpoint
SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbznSkYJS3RCCTReHRy4ex0vpSEI_a3IDbqvwNfybO0M3Quyf8bccbFS0F3WJtuyBpM0/exec'

myname = sys.argv[1]
myip = sys.argv[2]

# IP addresses to ping
names = ["Adam's MacBook Pro"]
ips = ['10.0.0.245']

def append_to_sheet(values):
    data = json.dumps(values)
    response = requests.post(SCRIPT_URL, data=data)
    if response.text == "Success":
        print("Data appended successfully")
    else:
        print("Error appending data")

# Number of pings to send
for i in range(1000):
    for (name, ip) in zip(names, ips):
        if ip == myip: # skip pinging myself
            continue

        # get a timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Ping command
        ping_cmd = 'ping -c 3 ' + ip

        # Run the ping command
        p = subprocess.Popen(ping_cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()

        # Parse the output
        lines = output.decode().split('\n')

        # Extract the latency from the last line
        latency = 0
        for line in lines:
            if 'time=' in line:
                latency = float(line.split('time=')[1].split(' ')[0])

        # Log the result
        data = [timestamp, myname, myip, name, ip, latency]
        append_to_sheet(data)

        # Wait for 10 seconds
        time.sleep(10)
    
    # Wait for 10 minutes between rounds
    time.sleep(600)

sys.exit(0)