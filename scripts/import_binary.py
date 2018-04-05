#!/usr/bin/env python
# Script to import binary file and expose structure
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


filename = "15_10_07_retina_dataset/naturalscene.h5"
f = h5py.File(filename, "r")

print("/")
keys = list(f.keys())
print(keys)
# ['spikes', 'test', 'train']

print("\n/spikes")
spikes = list(f["spikes"])
print(spikes)
# ['cell01', 'cell02', 'cell03', 'cell04', 'cell05', 'cell06', 'cell07', 'cell08', 'cell09']

print("\n/spikes/cell01")
cell01 = np.asarray(f["spikes"]["cell01"])
print("data shape:", cell01.shape)
# data shape: (43927,)

print("\n/test")
test = list(f["test"])
print(test)
# ['repeats', 'response', 'stimulus', 'time']

print("\n/test/repeats")
test_repeats = list(f["test"]["repeats"])
print(test_repeats)
# ['cell01', 'cell02', 'cell03', 'cell04', 'cell05', 'cell06', 'cell07', 'cell08', 'cell09']

print("\n/test/repeats/cell01")
repeats_cell01 = np.asarray(f["test"]["repeats"]["cell01"])
print("data shape:", repeats_cell01.shape)
# data shape: (5, 5997)

print("\n/test/response")
test_response = list(f["test"]["response"])
print(test_response)
# ['binned', 'firing_rate_10ms', 'firing_rate_20ms', 'firing_rate_5ms']

print("\n/test/response/binned")
test_response_binned = np.asarray(f["test"]["response"]["binned"])
print("data shape:", test_response_binned.shape)
# data shape: (9, 5997)

print("\n/test/response/firing_rate_10ms")
test_response_10ms = np.asarray(f["test"]["response"]["firing_rate_10ms"])
print("data shape:", test_response_10ms.shape)
# data shape: (9, 5997)

print("\n/test/stimulus")
test_stimulus = np.asarray(f["test"]["stimulus"])
print("data shape:", test_stimulus.shape)
# data shape: (5996, 50, 50)

# export all stimuli images
#for i in range(5996):
#	fig = plt.figure()
#	plt.gray()
#	plt.imshow(np.asarray(f["test"]["stimulus"])[i,:,:])
#	plt.savefig("test_stimuli/"+str(i)+".png")
#	plt.close()
#	print(i,end="\r")

print("\n/test/time")
test_time = np.asarray((f["test"]["time"]))
print("data shape:", test_time.shape)
# data shape: (5996,)

print("\n/train")
train = list(f["train"])
print(train)
# ['response', 'stimulus', 'time']
