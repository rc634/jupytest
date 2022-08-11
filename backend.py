

# Diarisation code with pyannote-audio https://github.com/pyannote/pyannote-audio
# Speech to text by Google https://realpython.com/python-speech-recognition/
# backend script by Robin Croft in transcript_generator.py


# this is only needed to time parts of the code
import time 

# needed to delete the modified audio file if created
import os

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

#for counting and collecting words spoken
from collections import defaultdict
import re

# class wrapper around diarization and speech to text, in file transcript_generator.py
from transcript_generator import TranscriptGenerator

obj=TranscriptGenerator("../private_jupyter/audio.wav",True,True,True,7.5*60.,9.5*60.)
obj.run()
obj.print_transcript()
obj.main_word_count(4)
obj.print_total_speaker_times(4)
obj.query_occurance(["take","you","no","why","how"],4)
obj.clean() # probably un-necesary
