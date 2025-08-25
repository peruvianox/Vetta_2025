import serial
import serial.tools.list_ports
import struct
import time
import json
import sys
import numpy as np
import scipy
from scipy import signal
import joblib
# import random
import os
from os.path import exists
from enum import Enum
# import onnx
# import onnxruntime
import warnings
# import torch
import tensorflow as tf

#For reasons related to serial interfaces in Unity, it is easiest for demo
#purposes to externalize
#communication with the Stryde hub to this process.  Anything related to the
#USB device, including detecting connection/removal
#should be in this process.  Status and data should be communicated via simple
#API and/or redirecting StandardIO from the Unity side.

# if needed, install hub driver for Hub: Adafruit HUZZAH32 - ESP32 Feather 
# https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=overview

#TODO: Fully embedded interpreter to allow graceful shutdown
#rather than killing the whole subprocess


# Constants

# Iterate This!

modelFile = 'Models/tfmodel_5.onnx'
# model = None

#Time-based (0) or an attempted real time construction (1)
StepDetectionMethod = 2

#Sensor unicode values
waistID = "38"
leftShankID = "39"
rightShankID = "3a"

#Sample lists
leftWaistSamples = []
rightWaistSamples = []
leftShankSamples = []
rightShankSamples = []

# Should be true on first event.  Set to false and samples are cleared on second event
#"Stance" is probably a misnomer at this point due to changes in event detection.
leftStance = False
rightStance = False

minShankSampleCount = 20

#Used for waveform id
leftPeakId = 0
rightPeakId = 0

sampleCount = 0

#VMJerk values: Initialize once
cutoff =6
gain = cutoff / np.sqrt(2)
sos = signal.butter(2, gain, fs =100, output='sos')

#Actually passes values to the model
def PredictPeakVGRF(waistSamples,id,side, model):

    global vgrfWaveForms

    #These parameters will have to be doublechecked
    height = .8
    distance = 10
    prominence = .15
    width = 2
    # if model == None:
    #     print("No Model Loaded")
    #     return

    magnitudes = []
    print('getting samples')
    # print(waistSamples)
    for sample in waistSamples:
        magnitudes.append(GetMagnitude(sample.accel))
    #print(len(magnitudes))
    print(magnitudes)
    inter_magnitudes = signal.resample(magnitudes,100)
    # inter_magnitudes = signal.resample_poly(magnitudes, 100, len(magnitudes))
    # print(len(inter_magnitudes))
    print(inter_magnitudes)
    # print(type(inter_magnitudes))
    # print(inter_magnitudes.shape)
    waist = list([float(x) for x in inter_magnitudes])
    print('input')

    input = tf.TensorSpec.from_numpy(np.asarray(waist))
    input.name = 'dense_24_input'
    print(input)
    # vgrf = model.predict([inter_magnitudes])[0]
    try:
    #     # vgrf = session.run(['dense_26'], {'dense_24_input': [np.float32([1]), 
    #     #                                                      np.asarray(waist, dtype=np.float32)]})
    #     vgrf = session.run(['dense_26'], {'dense_24_input': np.asarray(waist, dtype=np.float32)})

    #     # vgrf = session.run(['dense_26'], {'dense_24_input': np.asarray(waist, dtype=np.float32)})
        vgrf = model()
        print('output')
        print(vgrf)
    except Exception as e:
        print(f"{type(e)}: {e}")
    # T = torch.from_numpy(np.asarray(waist))
    # try:
    #     vgrf = session.run(None, {'input': T})
    # except:
    #     raise 'error in running model'
    #print(vgrf)

    #save full wave form
    vgrfWaveForm = VGRFWaveForm(id,time.time(),side,vgrf)
    jsonData = str(vgrfWaveForm)
    vgrfWaveForms.append(jsonData)

    #Grab peak vgrf for stimulus
    peaks,properties = signal.find_peaks(vgrf, height = height, prominence = prominence, width = width, distance = distance)
    peakSample = VGRFSample(id,time.time(),side,properties['peak_heights'][0])
    return peakSample

def LoadModel():
    return joblib.load(modelFile) 

def GetMagnitude(sample):
    return np.sqrt(sample[0] ** 2 + sample[1] ** 2 + sample[2] ** 2)

#Derived from Ricky's example
def VectorMagJerk(samples):
    global sos
    x = []
    y = []
    z = []
    for sample in samples:
        x.append(sample.accel[0])
        y.append(sample.accel[1])
        z.append(sample.accel[2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    shankValues = np.array([x,y,z])
    #Vector Norm Jerk
    jerk = np.diff(np.linalg.norm(shankValues,axis=0))

    F = signal.sosfiltfilt(sos,shankValues.T, axis = 0)
    #Filt = pd.DataFrame(data=F)
    xF = F[0]
    yF = F[1]
    zF = F[2]

    VMF =np.sqrt(np.multiply(xF,xF) + np.multiply(yF,yF) + np.multiply(zF,zF)).tolist()

    
    return VMF

def GetVMAJ(samples):
    # return vector magnitude acceleration and jerk from list-based input
    #global SampleCols 
    
    #Sam = np.reshape(samples, (len(samples), len(SampleCols)))
    #x = np.array(Sam[:, 2])
    #y = np.array(Sam[:, 3])
    #z = np.array(Sam[:, 4])
    #Values = np.array([x,y,z])

    x = []
    y = []
    z = []
    for sample in samples:
        x.append(sample.accel[0])
        y.append(sample.accel[1])
        z.append(sample.accel[2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    shankValues = np.array([x,y,z])


    VMA = np.linalg.norm(shankValues,axis=0)
    jerk = np.diff(VMA)     # Vector Norm Jerk
    
    return VMA, jerk

def FindHeelStrikes(jerk):
    ht = 3
    dis = 50
    IClocs, ICprops = signal.find_peaks(jerk, height=ht, distance=dis)
    ICpks = [jerk[x] for x in IClocs]

    return IClocs, ICpks


#Currently no references.  Initial and Final event detection.  FindHeelStrikes is what's currently used
def FindGaitEvents(jerk):
    prom = 5 # specify promimence for small peak
    [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
    FCpks = [VMF[x] for x in FClocs]
    
    # get initial contact times
    prom = (1, 5)  # specify promimence for large peak
    wid = (5, 20)
    [IClocs, ICprops] = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
    ICpks = [VMF[x] for x in IClocs]

    return FCpks,ICpks



#def FindHeelStrikes(VMF):
#    prom = (.5, 4)  # specify promimence for large peak
#    wid = (5, 30)
#    IClocs, ICprops = signal.find_peaks(np.multiply(-1, VMF), prominence=prom, width=wid)
#    ICpks = [VMF[x] for x in IClocs]
#    #peaks,_ = scipy.signal.find_peaks(jerk,height = 5,distance=10)
#    return ICpks

#Need to test height band parameters tomorrow morning.
def FindToeOffs(VMF):
    prom = 5 # specify promimence for small peak
    [FClocs, FCprops] = signal.find_peaks(VMF, prominence=prom)
    FCpks = [VMF[x] for x in FClocs]
    return FCpks


class VGRFWaveForm:
    def __init__(self):
        self.id = -1
        self.time = -1
        self.side = ""
        self.values = []
    def __init__(self,new_id,new_time,new_side,new_values):
        self.id = new_id
        self.time = new_time
        self.side = new_side
        self.values = new_values
    def __str__(self):
        return str(self.id) +","+str(self.time)+","+str(self.side)+","+str(self.values).replace("[","").replace("]","")

class VGRFSample:
    def __init__(self):
        self.id = -1
        self.time = -1
        self.side = ""
        self.peakValue = -1
    def __init__(self,new_id,new_time,new_side,new_peakValue):
        self.id = new_id
        self.time = new_time
        self.side = new_side
        self.peakValue = new_peakValue

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


targetDeviceString = "Silicon Labs CP210x USB to UART Bridge"
serial_port = FindTargetDevice(targetDeviceString)

if (serial_port != None):
    targetDevicePresent = True
    hubStatus = HubStatus.CONNECTED
    print("Hub Connected!")
    serial_port.write(b'S')  # start recording command into serial port
else:
    targetDevicePresent = False
    hubStatus = HubStatus.DISCONNECTED

# How frequently to send a new HubSample in whole seconds
hubUpdateTime = 1
lastHubUpdate = time.time()

lastSensorSampleTime = -1
connectedSensors = set()
sensorSampleTimes = {}
# In whole seconds
sensorTimeout = .3
connectionStablizeDuration = 10
connectionStabilizeTime = -1

line = []
last_byte = ''
newPacket = False
running = True
maxSamples = 20
count = 0

#For saving to a file
samples = []
vgrfSamples = []
vgrfWaveForms = []
testVRGFId = 0
testVGRFDelta = .5
testLastVGRFUpdate = -1
side = "Left"

#Load the Model
model = tf.saved_model.load('Models/tfmodel_5')

lStepTime = -1
rStepTime = -1
samples_gait_event = 100
shankSampleTarget = 100

while running:
    try:
     
        #If the device is connected, read data
        if targetDevicePresent:
            for c in serial_port.read():
                if newPacket:
                    line.append(hex(c))
                    if c == 204 and last_byte == 170:
                        newPacket = True
                        line = []
                        line.append(hex(last_byte))
                        line.append(hex(c))
                    
                    #A new packet/ End of the packet is detected
                    elif c == 51 and last_byte == 85:
                        #print("Line: ",line)
                        newPacket = False
                        count+=1
                        #print(time.time())
                        #Convert bytes to object
                        try:
                            new_sample = ProcessPacket(line)
                        except:
                            print("Error Creating Sample")
                            continue

                        #Create Shank Signals for step counting
                        #print(new_sample.id)
                        #print(leftShankId)
                        if new_sample.id == leftShankID:
                            if len(leftShankSamples)>300:
                                leftShankSamples = []
                                leftWaistSamples =[]
                            leftShankSamples.append(new_sample)
                        elif new_sample.id == rightShankID:
                            if len(leftShankSamples)>300:
                                rightShankSamples = []
                                rightWaistSamples =[]
                            rightShankSamples.append(new_sample)

                        #Create Waist Signals for predictions
                        if new_sample.id == waistID:
                            #if leftStance:
                            leftWaistSamples.append(new_sample)
                            #if rightStance:
                            rightWaistSamples.append(new_sample)
                        
                        sampleCount+=1
                        
                        # Update sensor status metrics
                        connectedSensors.add(new_sample.id)
                        sensorSampleTimes[new_sample.id] = new_sample.time
                        lastSensorSampleTime = new_sample.time

                        # print(sampleCount)
                        print(new_sample)
                        # print(len(line))

                        if sampleCount > 20:
                            raise StopIteration
                        
                        # StepCheck
                        print('Detecting Steps ...')
                        if StepDetectionMethod == 0:
                            #This method hasn't been updated in some time.
                            if len(leftShankSamples) >= shankSampleTarget:
                                # jerk = VectorNormJerk(leftShankSamples)
                                jerk = VectorMagJerk(leftShankSamples)
                                steps = FindHeelStrikes(jerk)
                                print("Left Step Count:" + str(len(steps)))
                                leftShankSamples = []

                            if len(rightShankSamples) >= shankSampleTarget:
                                # jerk = VectorNormJerk(rightShankSamples)
                                jerk = VectorMagJerk(rightShankSamples)
                                steps = FindHeelStrikes(jerk)
                                print("Right Step Count:" + str(len(steps)))
                                rightShankSamples = []
                        
                        
                        if StepDetectionMethod == 1:
                            if leftStance:

                                # Find second heelstrike, predict VGRF, send to STDOut, and save
                                # Modified this to be heelstrike to heelstrike
                                if len(leftShankSamples) >= minShankSampleCount:
                                     jerk = VectorMagJerk(leftShankSamples)
                                     steps = FindHeelStrikes(jerk)
                                     if (len(steps) > 0):
                                            print("Left Ending Heelstrike!")
                                            leftStance = False
                                            leftShankSamples = []
                                            #Predict VGRF and send sample
                                            #Use only 60% of the sample signal
                                            stopIndex = int(len(leftWaistSamples)*.6)+1
                                            with warnings.catch_warnings():
                                                warnings.simplefilter("ignore")
                                                leftVGRFSample = PredictPeakVGRF(leftWaistSamples[:stopIndex],leftPeakId,"Left")
                                                jsonData = json.dumps(leftVGRFSample.__dict__)
                                            print(jsonData,end='\n')
                                            leftWaistSamples = []
                                            leftPeakId+=1
                                            vgrfSamples.append(jsonData)

                            #Find first heel strike 
                            elif len(leftShankSamples) >= minShankSampleCount:
                                jerk = VectorMagJerk(leftShankSamples)
                                steps = FindHeelStrikes(jerk)
                                if (len(steps) > 0):
                                    print("Left Step!")
                                    leftStance = True
                                    leftShankSamples = []
                            
                            #Find second heelstrike, predict VGRF, send to STDOut, and save
                            if rightStance:
                                #Modified this to be heelstrike to heelstrike
                                if len(rightShankSamples) >= minShankSampleCount:
                                     jerk = VectorMagJerk(rightShankSamples)
                                     steps = FindHeelStrikes(jerk)
                                     if (len(steps) > 0):
                                            print("Right Ending Step!")
                                            rightStance = False
                                            rightShankSamples = []
                                            #Predict VGRF and send sample
                                            #Use only 60% of the sample signal
                                            stopIndex = int(len(rightWaistSamples)*.6)+1
                                            rightVGRFSample = PredictPeakVGRF(rightWaistSamples[:stopIndex],rightPeakId,"Right")
                                            jsonData = json.dumps(rightVGRFSample.__dict__)
                                            print(jsonData)
                                            rightWaistSamples = []
                                            rightPeakId+=1
                                            vgrfSamples.append(jsonData)
                            
                            #Find first heel strike 
                            elif len(rightShankSamples) >= minShankSampleCount:
                                jerk = VectorMagJerk(rightShankSamples)
                                steps = FindHeelStrikes(jerk)
                                if (len(steps) > 0):
                                    print("Right Step!")
                                    rightStance = True
                                    rightShankSamples = []
                        
                        if StepDetectionMethod == 2:
                            if len(leftShankSamples) >= shankSampleTarget: # if sufficient samples to check for a gait event
                                print('sufficient samples')
                                VMA, jerk = GetVMAJ(leftShankSamples)
                                Lind, pks = FindHeelStrikes(jerk)

                                if len(Lind) > 0:
                                    #LStepTimes.append(leftShankSamples[Lind[0]][1])
                                    print("Left Step Found!")
                                    if not leftStance:
                                        #print("First Step!")
                                        leftStance = True
                                        leftWaistSamples = []
                                        lStepTime = time.time()
                                    else:
                                        currentTime = time.time()
                                        if currentTime-lStepTime<2000:
                                            print("Ending Step!")
                                            stopIndex = int(len(leftWaistSamples)*.6)+1
                                            with warnings.catch_warnings():
                                                warnings.simplefilter("ignore")
                                                print('Predicting vGRF')
                                                leftVGRFSample = PredictPeakVGRF(leftWaistSamples[:stopIndex],leftPeakId,"Left", model)
                                                print('Predicting vGRF')
                                                jsonData = json.dumps(leftVGRFSample.__dict__)
                                                print(jsonData)
                                                #leftWaistSamples = []
                                                vgrfSamples.append(jsonData)
                                                leftPeakId+=1
                                            #Get Waist Samples to Model
                                        #else:
                                            #print("Orphaned Step! Discarding!")
                                        leftStance = False
                                leftShankSamples = []
                            if len(rightShankSamples) >= shankSampleTarget: # if sufficient samples to check for a gait event
                                VMA, jerk = GetVMAJ(rightShankSamples)
                                Rind, pks = FindHeelStrikes(jerk)

                                if len(Rind) > 0:
                                    #LStepTimes.append(leftShankSamples[Lind[0]][1])
                                    #print("Right Step Found!")
                                    if not leftStance:
                                        #print("First Step!")
                                        rightStance = True
                                        rightWaistSamples = []
                                        rStepTime = time.time()
                                    else:
                                        currentTime = time.time()
                                        if currentTime-rStepTime<2000:
                                            stopIndex = int(len(rightWaistSamples)*.6)+1
                                            with warnings.catch_warnings():
                                                warnings.simplefilter("ignore")
                                                rightVGRFSample = PredictPeakVGRF(rightWaistSamples[:stopIndex],rightPeakId,"Right", session)
                                                jsonData = json.dumps(rightVGRFSample.__dict__)
                                                print(jsonData)
                                                rightPeakId+=1
                                                vgrfSamples.append(jsonData)
                                            
                                            #print("Ending Step!")
                                            #Get Waist Samples to Model
                                        #else:
                                        #    print("Orphaned Step! Discarding!")
                                        rightStance = False
                                rightShankSamples = []

                        #        if len(LStepTimes) > 2:
				                    ## get previous two heel strike events
                        #            Curr = LStepTimes[-1]
                        #            Prev = LStepTimes[-2]

                        #            if Curr - Prev < timeThresh: # ensure they are no more than 2 sec apart
      
                        #                # get waist samples during that time
                        #                GCStart = Waist[Waist['time'] == Prev].index[0]
                        #                GCEnd = Waist[Waist['time'] == Curr].index[0]
                    
                        #                # get VMA for Waist
                        #                x = Waist['accel x'][GCStart:GCEnd]  
                        #                y = Waist['accel y'][GCStart:GCEnd]  
                        #                z = Waist['accel z'][GCStart:GCEnd]  
                        #                W_VMA = np.linalg.norm(np.array([x,y,z]),axis=0)
                
                        #                # resample to 100 points
                        #                ReSamWaist = signal.resample(W_VMA, 100)

                        # Save sensor json

                        #if (sampleCount == 8):
                        jsonData = json.dumps(new_sample.__dict__)
                        #Send to STDOut
                        #NOT NEEDED ANYMORE
                        #print(jsonData)
                        sampleCount = 0
                        samples.append(jsonData)
                        line = []

                elif c == 204 and last_byte == 170:
                    newPacket = True
                    line.append(hex(last_byte))
                    line.append(hex(c))
                last_byte = c

        # await a matching USB device connection
        else:
            # Sleep thread a set time
            # time.sleep(1)
            # Check for an attached device
            serial_port = FindTargetDevice(targetDeviceString)
            if (serial_port != None):
                targetDevicePresent = True
                hubStatus = HubStatus.CONNECTED
    
            # Send a status update
            if (targetDevicePresent):
                print("Hub Connected!")
            else:
                print("Waiting for device!")
        
        cur_time = time.time()
        # Check For Low Connectivity/Dropped Sensors
        for key in sensorSampleTimes:
            if cur_time - sensorSampleTimes[key] > sensorTimeout:
                if key in connectedSensors:
                    connectedSensors.remove(key)
        if hubStatus != HubStatus.DISCONNECTED:
            if len(sensorSampleTimes) != len(connectedSensors):
                hubStatus = HubStatus.LOW_CONNECTIVITY
            else:
                if connectionStabilizeTime == -1:
                    connectionStabilizeTime = cur_time
                elif cur_time - connectionStabilizeTime >= connectionStablizeDuration:
                    connectionStabilizeTime = -1
                    hubStatus = HubStatus.CONNECTED


        # Send HubSample
        if cur_time - lastHubUpdate >= hubUpdateTime:
            jsonData = json.dumps(HubSample(cur_time,hubStatus,lastSensorSampleTime,sorted(connectedSensors)).__dict__)
            print(jsonData)
            #print(HubSample(cur_time,hubStatus,lastSensorSampleTime,connectedSensors))
            lastHubUpdate = cur_time
                
    except KeyboardInterrupt:
        running = False
        pass
    except serial.SerialException as e:
        # There is no new data from serial port
        print("Serial Exception!")
        targetDevicePresent = False
        hubStatus = HubStatus.DISCONNECTED
        pass
    except BaseException as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        running = False
        pass


print("Saving Recorded Sensor Data")
# save raw data
fileCount = "new_Raw"
savePath = 'Output/' + str(fileCount) + ".json"
counter = 0
while exists(savePath):
    counter+=1
    savePath = savePath.replace(".json","_"+str(counter)+".json")
with open(savePath, "w") as f:
    f.write(json.dumps(samples,indent=4, sort_keys=True))
# save step loading peaks
stepsFile = "new_Steps"
savePath = 'Output/' + stepsFile + ".json"
counter = 0
while exists(savePath):
    counter+=1
    savePath = savePath.replace(".json","_"+str(counter)+".json")
with open(savePath, "w") as f:
    f.write(json.dumps(vgrfSamples,indent=4, sort_keys=True))
# save vGRF waveforms
waveFormFile = "new_WaveForms"
savePath = 'Output/' + waveFormFile + ".json"
counter = 0
while exists(savePath):
    counter+=1
    savePath = savePath.replace(".json","_"+str(counter)+".json")
with open(savePath, "w") as f:
    f.write(json.dumps(vgrfWaveForms,indent=4, sort_keys=True))

serial_port.close()
