##### Voice Data Preparation for Voice Recognition - Machine learning on edge device

Python scripts to generate sample data from a continuous voice-stream (e.g. numbers from 0 to 9).
To reduce effort on recording data, the scripts generate multiple versions of this samples by shifting start/end, random gain and low-pass filter 
In case of different labels in one file, it uses Google-Speech-To-Text API to automatically label the data.
Finally, the files are saved as separate samples, e.g. to be uploaded to EdgeImpulse.

###### Script: voice_acquisition
This scripts records a raw binary soundfile from any device by running a web-server.
The edge-device needs to post the data as raw audio stream.
There is no limitation on length, file is recorded until server is stopped.

###### Script: voice_preparation
The script takes a long input file, with multiple samples (e.g. repeating allways the same keyword - label)
After silent detection this file is separated in chunks, each chunk representing a single word/number.
Depending on settings, the label is classified via calling Goole-Speech-To-Text api or taken as default value.

**To ease the generation of multiple samples, it generates various versions of test-data from one chunk by:**
* slightly changing start and end (within silence window)
* adjusting gain by random
* adding low-pass filter by random



