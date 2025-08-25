import serial
import time
from enum import Enum
import struct


class Sample:
    def __init__(self):
        self.id = -1
        self.time = -1
        self.accel = []
        self.gyro = []
        self.mag = []
        self.flag = -1
        self.battery = -1
        self.battery_percent = -1
    def __init__(self, new_id, new_time, new_accel, new_gyro, new_mag, new_flag, new_battery, battery_percent):
        self.id = new_id
        self.time = new_time
        self.accel = new_accel
        self.gyro = new_gyro
        self.mag = new_mag
        self.flag = new_flag
        self.battery = new_battery
        self.battery_percent = battery_percent
    def __str__(self):
        return 'Sensor: ' + str(self.id) + '\nAccel: ' + str(self.accel[0]) + ',' + str(self.accel[1]) + ',' + str(self.accel[2]) + '\nGyro: ' + str(self.gyro[0]) + ',' + str(self.gyro[1]) + ',' + str(self.gyro[2]) + '\nMag: ' + str(self.mag[0]) + ',' + str(self.mag[1]) + ',' + str(self.mag[2]) + '\nStatus: ' + str(self.flag)

class HubSample:
    def __init__(self):
        self.time = -1
        self.connectionStatus = HubStatus.DISCONNECTED
        self.lastSensorSampleTime = -1
        self.connectedSensors = []
    def __init__(self,new_time,new_status,new_SampleTime,new_sensors):
        self.time = new_time
        self.connectionStatus = new_status
        self.lastSensorSampleTime = new_SampleTime
        self.connectedSensors = new_sensors

class HubStatus(int,Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    LOW_CONNECTIVITY = 2

def GetSerialPorts():
    print("Looking for Serial Devices")
    myports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
    return myports

def FindTargetDevice(target):
    ports = GetSerialPorts()
    for candidate in ports:
        if target in candidate[1]:
            print(candidate[0])
            #Construct a Serial Object using the found port id
            serial_port = serial.Serial(port=str(candidate[0]),\
                baudrate=230400,\
                parity=serial.PARITY_NONE,\
                stopbits=serial.STOPBITS_ONE,\
                bytesize=serial.EIGHTBITS,\
                timeout=0)
            return serial_port
    return None
    
#Read an individual packet from the hub and convert it to Sample instance
def ProcessPacket(packetBin):
    # timestamp
    new_time = time.time()

    # Decode Sensor ID
    # This assumes less than 8 sensors to cut down on conversionss
    new_id = str(packetBin[2][2:])
    #print(new_id)

    # Decode Accel, Gyro, and Mag as 3-element float tuples
    new_accel = []
    new_gyro = []
    new_mag = []

    # Accel
    for x in range(3):
        start = 4 + (4 * x)
        new_float = bytes()
        for y in range(start,start + 4):
            hex_string = packetBin[y][2:]
            #print(hex_string)
            if len(hex_string) < 2:
                hex_string = '0' + hex_string
            new_float+=(bytes.fromhex(hex_string))
        #print("\n")
        value = struct.unpack('f', (new_float))[0]
        new_accel.append(value)
    #print(new_accel)

    # Gyro
    for x in range(3):
        start = 16 + (4 * x)
        new_float = bytes()
        for y in range(start,start + 4):
            hex_string = packetBin[y][2:]
           #print(hex_string)
            if len(hex_string) < 2:
                hex_string = '0' + hex_string
            new_float+=(bytes.fromhex(hex_string))
        #print("\n")
        value = struct.unpack('f', (new_float))[0]
        new_gyro.append(value)
    #print(new_gyro)

    # Mag
    for x in range(3):
        start = 28 + (4 * x)
        new_float = bytes()
        for y in range(start,start + 4):
            hex_string = packetBin[y][2:]
            #print(hex_string)
            if len(hex_string) < 2:
                hex_string = '0' + hex_string
            new_float+=(bytes.fromhex(hex_string))
        #print("\n")
        value = struct.unpack('f', (new_float))[0]
        new_mag.append(value)
    #print(new_mag)

    # Flag
    try:
        new_flag = int(packetBin[40][2:])
    except ValueError:
        new_flag = 0
    #print(new_flag)

    # Battery
    new_float = bytes()
    hex_string = packetBin[41][2:]
    new_battery = int(hex_string, 16)
    max_battery = 220
    battery_percent = round(new_battery / max_battery, 3)
    
    # package updates into sample class
    new_sample = Sample(new_id, new_time, new_accel, new_gyro, new_mag, new_flag, new_battery, battery_percent)
    # print(new_sample)
    # print('\n')
    return new_sample
