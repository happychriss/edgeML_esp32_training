from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.silence import detect_silence
import numpy as np
from pydub.playback import play
from google.cloud import speech
import io
import os
import shutil
import random
import string
import random
from datetime import datetime

random.seed(a=None, version=2)

RUN_NAME = "v1"
SOURCE_FOLDER = "./data/raw_data/"  # first character of filename = label, "G" use Google, X for ignore
TARGET_FOLDER = "./data/training_data"
SAMPLE_LENGTH = 800  # ms
SAMPLE_SAVE = 50  # to have some buffer on silet detection
MAX_VERSIONS_PER_NUMBER = 10  # how many different versions of a number will be created

### Google speech recognition, to detect mixed files
result_set = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
speech_context = speech.SpeechContext(phrases=['$OPERAND'])
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="de-DE",
    speech_contexts=[speech_context]
)
client = speech.SpeechClient()


class Sample():
    "Stores name and place pairs"

    def __init__(self, silent_start, sound_start, sound_end, silent_end, label):
        self.silent_start = silent_start
        self.sound_start = sound_start
        self.sound_end = sound_end
        self.silent_end = silent_end
        self.label = label


class Data():
    def __init__(self, idx, start, end, low_pass, db, label):
        self.idx = idx
        self.start = start
        self.end = end
        self.low_pass = low_pass  # low pass used
        self.db = db  # change in DB used
        self.label = label


def process_file(filename):
    sound = AudioSegment.from_raw(filename, sample_width=2, frame_rate=16000, channels=1)

    print("************** Silence Detection **************")

    chunks = detect_silence(
        sound,
        # split on silences longer than 1000ms (1 sec)
        min_silence_len=200,
        # anything under -16 dBFS is considered silence
        silence_thresh=-51,

    )

    print("Silence Chunks:")
    print(chunks)

    print("************** Build Orginal numbers + Label:" + initial_data_label)

    # # build orginal number array #  [-1/0]ssssss[-1/1]mmmmmm[0/0]sssss[0/1]
    org_numbers = []
    for idx, chunk in enumerate(chunks):
        if idx > 1:
            org_number = []
            org_number.append(chunks[idx - 1][1])
            org_number.append(chunks[idx][0])

            silent_start = chunks[idx - 1][0] + SAMPLE_SAVE
            sound_start = chunks[idx - 1][1] - SAMPLE_SAVE
            sound_end = chunks[idx][0] + SAMPLE_SAVE
            silent_end = chunks[idx][1] - SAMPLE_SAVE
            # print(str(silent_start) + "-" + str(sound_start) + "-" + str(sound_end) + "-" + str(silent_end))

            # set label using google speech or default
            if initial_data_label == "G":
                data_label = ""
                audio = speech.RecognitionAudio(content=sound[sound_start:sound_end].raw_data)
                response = client.recognize(config=config, audio=audio)
                if response:
                    data_label = response.results[0].alternatives[0].transcript
                    print("Google-Idx" + str(idx) + "identified as: " + data_label)

                else:
                    print("Error not recognized sound with idx " + str(idx) + "saved to file")
                    sound[sound_start:sound_end].export(out_f="s_ERROR_" + str(idx) + ".wav", format='wav')
                    break

            else:
                data_label = initial_data_label

            org_numbers.append(Sample(silent_start, sound_start, sound_end, silent_end, data_label))

    print("******************+ Build  different versions *********")
    vers_numbers = []

    for idx, o_num in enumerate(org_numbers):

        si_length_1 = o_num.sound_start - o_num.silent_start
        si_length_2 = o_num.sound_end - o_num.sound_start
        sound_lenght = o_num.sound_end - o_num.sound_start

        max_silent = min([si_length_1, si_length_2, (SAMPLE_LENGTH - sound_lenght)])  # that is my range to shift start/end of signal

        samples_num = int(max_silent / 20)
        print("Wave(" + str(idx) + "): " + str(samples_num), end=" ")
        if samples_num > MAX_VERSIONS_PER_NUMBER:
            samples_num = MAX_VERSIONS_PER_NUMBER
            print(" reduced to: " + str(MAX_VERSIONS_PER_NUMBER), end=" ")
        print("")

        for i in range(1, samples_num):
            sound_new_start = o_num.sound_start - random.randrange(20, max_silent)
            sound_new_end = sound_new_start + SAMPLE_LENGTH

            vers_number = []

            vers_number.append(idx)
            vers_number.append(sound_new_start)
            vers_number.append(sound_new_end)

            vers_numbers.append(Data(
                idx,
                sound_new_start,
                sound_new_end,
                8000,  # low pass
                0,
                o_num.label
            ))  # db

            vers_numbers.append(Data(
                idx,
                sound_new_start,
                sound_new_end,
                random.randrange(500, 4000),  # lowpass
                random.randrange(-15, 2),  # gain
                o_num.label
            ))

    print("Number of different versions created:" + str(len(vers_numbers)))
    print("**************** Generating Training-Data files exporting to:" + TARGET_FOLDER)

    # export as wav
    for idx, number in enumerate(vers_numbers):
        output = (sound[number.start:number.end] + number.db).low_pass_filter(number.low_pass)
        #    play(output)
        filename = number.label + "." + RUN_NAME + "_" + str(number.idx) + "_DB" + str(number.db) + "_LP" + str(
            number.low_pass) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        print(filename)
        filename = os.path.join(TARGET_FOLDER, filename)
        output.export(out_f=filename, format='wav')

    print("DONE")


# START PROGRAM **************************************************************************************************

# Prepare file list **************************************************************************************************
# G=Label via Google, X=ignore, file should end with raw

initial_data_label = ""
for entry in os.scandir(SOURCE_FOLDER):
    if entry.name.endswith(".raw") and entry.name[0] != "X":
        if entry.name[0] in result_set:
            initial_data_label = str(entry.name[0])
            print("!!!!!!!!!!!!!!!!! Processing file:" + entry.name + " with label: " + initial_data_label + " !!!!!!!!!!!!!!!!!!!!")
            process_file(entry.path)
        else:
            print("Invalid file:" + entry.name)
