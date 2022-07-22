

# Diarisation code with pyannote-audio https://github.com/pyannote/pyannote-audio
# Speech to text by Google https://realpython.com/python-speech-recognition/
# backend script by Robin Croft


# this is only needed to time parts of the code
import time 

#for editing wav files, for example cutting them into smaller clips
from pydub import AudioSegment 
from pydub.playback import play
from pydub.utils import mediainfo

#for speech to text
import speech_recognition as sr

#for pyannote-audio's diarisation
import torch
from huggingface_hub import HfApi
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

audio = "filename.wav" #need to specify this

# time window to diarize, shorter is faster, longer is more accurate. Time is in seconds
GLOBAL_START = 0.
GLOBAL_END = 60. 

# make a new wav file that is the window specified by GLOBAL_START and GLOBAL_END
wav_file = AudioSegment.from_file(file = "audio.wav", format = "wav")
info = mediainfo("audio.wav")
ratio = float(len(wav_file))/float(info["duration_ts"])
sample_rate = wav_file.frame_rate*ratio
start_index = int(GLOBAL_START*sample_rate)
end_index = int(GLOBAL_END*sample_rate)
modified_wav_file = wav_file[start_index:end_index]
modified_wav_file.export(out_f = "segment.wav" , format = "wav")



# speech to text on whole file
r = sr.Recognizer()
audio_sr = sr.AudioFile("segment.wav")
start_time = time.time()
with audio_sr as source:
    audiodata = r.record(source)
    #audiodata = r.record(source, offset=START_TIME, duration = END_TIME-START_TIME) # could look at a small segment like this
try:
    print(r.recognize_google(audiodata,language="en-GB"))
except Exception as e:
    print("Error : " + str(e))
end_time = time.time()
print("\nTime taken : ",end_time-start_time)






# diarization on file between
start_time = time.time()
dia = pipeline("segment.wav")
end_time = time.time()
end_time = time.time()
print("\nTime taken : ",end_time-start_time)


# print out diarisation
for turn, _, speaker in dia.itertracks(yield_label=True):
	print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")