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


# a_variable is an argument
# self.variable is a class member
# None is the null object
# filenames must be like "audio.wav"
# a_verbose controlls wether to print lots of information in the main() function
# a_perform_controls wether to print how long the diarization takes aswell as the generation of the transcript
class TranscriptGenerator:
    def __init__(self, a_filename, a_perform_timing=False, a_verbose=False, a_do_file_cutting=False, a_start_time=None, a_end_time=None):
        self.filename = a_filename
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        self.dia = None
        self.modified_filename = None
        self.start_time = a_start_time
        self.end_time = a_end_time
        self.do_file_cutting = a_do_file_cutting
        self.merged_fragments = []
        self.speech_fragments = []
        self.transcript = []
        self.perform_timing = a_perform_timing
        self.verbose = a_verbose 
        self.spkr_counts = None
        self.speaker_times = None

        # safety checks
        if (a_do_file_cutting): 
            if(a_start_time==None or a_end_time==None): 
                print("\nerror: Check input times for file cutting!")
        
        if a_filename[-4:] != ".wav" :
            print("\nerror : check input file is a wav file with .wav extension!")

    # uses pydub to cut the audio file into a new file with start and end times
    # cut audiofile saved over a_modified_filename
    def cut_audio_file(self, a_filename, a_start_time=None, a_end_time=None):

        #load default params
        if (a_start_time == None): a_start_time=self.start_time
        if (a_end_time == None): a_end_time=self.end_time

        wav_file = AudioSegment.from_file(file = a_filename, format = "wav")
        info = mediainfo(a_filename)
        ratio = float(len(wav_file))/float(info["duration_ts"])
        sample_rate = wav_file.frame_rate*ratio
        start_index = int(a_start_time*sample_rate)
        end_index = int(a_end_time*sample_rate)
        modified_wav_file = wav_file[start_index:end_index]
        self.modified_filename = a_filename[0:-4] + "modified.wav" #assumed a_filename ends in .wav
        modified_wav_file.export(out_f = self.modified_filename , format = "wav") 

  
    # defaults to entire file, but can overwrite and specify times like
    # speech_to_text(filename,False,0,60) to use only the fist 60 seconds.
    def do_and_print_speech_to_text(self, a_filename, a_whole_file=True, a_start_time=0., a_end_time=0.):
        # speech to text on whole file
        time1 = time.time()
        r = sr.Recognizer()
        audio_sr = sr.AudioFile(a_filename)
        with audio_sr as source:
            if a_whole_file:
                audiodata = r.record(source)
            else : 
                audiodata = r.record(source, offset=a_start_time, duration = a_end_time-a_start_time) 
        try:
            print("\n\n ##### Speech to Text #####")
            print(r.recognize_google(audiodata,language="en-GB"))
        except Exception as e:
            print("Error : " + str(e))

        time2 = time.time()
        if self.perform_timing: 
            if not a_whole_file:
                print(f"\n>>>> Speech to text of " + a_filename + f" between {round(a_start_time,1)}s and {round(a_end_time,1)}s took {round(time2-time1,1)} seconds.")
            else :
                print(f"\n>>>> Speech to text of " + a_filename + f" took {round(time2-time1,1)} seconds.")

    # diarization done on whole file, if this is too long then try cutting the file with the cut_audio_file function
    # a_perform_timing controlls wether we want to time the diarisation
    def do_diarization(self, a_filename):
        time1 = time.time()
        self.dia = self.pipeline(a_filename)
        time2 = time.time()
        if self.perform_timing: print(f"\n>>>> Diarization of " + a_filename + f" took {round(time2-time1,1)} seconds.")

    def merge_diarization(self):
        for turn, track, speaker in self.dia.itertracks(yield_label=True):
            if (True):#turn.end > GLOBAL_START and turn.start < GLOBAL_END):
                self.speech_fragments += [[speaker,turn.start,turn.end,"new"]]
                #print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

        for i in reversed(range(len(self.speech_fragments)-1)): 
            frag = self.speech_fragments[i]
            nextfrag = self.speech_fragments[i+1]
            if frag[0]==nextfrag[0]:
                nextfrag[1] = frag[1]
                frag[2]=nextfrag[2]
                nextfrag[3] = "repeat"

        for frag in self.speech_fragments:
            if (frag[3]=="new"):
                self.merged_fragments += [frag]

    def generate_transcript(self,a_filename):
        if self.merged_fragments == [] : print("\nerror : must perform merge_diarization() before generating transcript")
        time1 = time.time()
        r = sr.Recognizer()
        audio_sr = sr.AudioFile(a_filename)
        for frag in self.merged_fragments:
            with audio_sr as source:
                    audiodata = r.record(source, offset=frag[1]-0.1, duration = frag[2]-frag[1]+0.1)
            try:
                words = r.recognize_google(audiodata,language="en-GB")
                self.transcript += [[frag[0],words]]
            except Exception as e:
                self.transcript += [[frag[0],"???"]]
        time2 = time.time()
        if self.perform_timing: print(f"\n>>>> Generating Transcript took {round(time2-time1,1)} seconds.")

    def print_transcript(self):
        if self.transcript==[]:print("\nerror : must generate transcript with generate_transcript() before printing it! ")
        print("\n\n##### Transcript #####\n")
        for line in self.transcript:
            print(line[0] + ' : ' + line[1] + '\n')

    def print_diarization(self):
        if self.dia :
            if self.merged_fragments==[]:
                # print out diarisation
                print("\n\n##### Diarization #####")
                for turn, _, speaker in self.dia.itertracks(yield_label=True):
                    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            else:
                # print out diarisation
                print("\n\n##### Merged Diarization #####")
                for frag in self.merged_fragments: print(f"start={frag[1]:.1f}s stop={frag[2]:.1f}s speaker_{frag[0]}")
        else :
            print("\nerror : must perform diarization with do_diarization() before printing it! ")

    def main_word_count(self,a_num_speakers=2):
        self.spkr_counts = [defaultdict(int) for i in range(a_num_speakers)]
        r = sr.Recognizer()

        if self.modified_filename : audio_sr = sr.AudioFile(self.modified_filename)
        else : 
            print("\nerror : must use merge_diarization() before word count!")
            return

        for frag in self.merged_fragments:
            with audio_sr as source:
                audiodata = r.record(source, offset=frag[1]-0.1, duration = frag[2]-frag[1]+0.1)
            try:
                words = r.recognize_google(audiodata,language="en-GB")
                for i in range(a_num_speakers):
                    if frag[0] == f'SPEAKER_0{i}':
                        for word in re.findall('\w+', words.replace("'"," ")):
                            self.spkr_counts[i][word] += 1
            except Exception as e:
                pass
        for i in range(a_num_speakers):
            print(f"\n\n##### Speaker {i} #####")
            print(self.spkr_counts[i])

    #outputs occurances of words in the list a_word_list
    def query_occurance(self,a_word_list,a_num_speakers=2):
        if self.modified_filename : audio_sr = sr.AudioFile(self.modified_filename)
        else : 
            print("\nerror : must use merge_diarization() before querying word occurance!")
            return
        if self.spkr_counts == None: 
            print("\nerror : must use main_word_count() before querying word occurance!")
            return
        
        for a_word in a_word_list:
            print(f"\n\n##### Occurance of \"{a_word}\" #####")
            for i in range(a_num_speakers):
                print(f"\n>>>>Speaker {i} says \"{a_word}\" {self.spkr_counts[i][a_word]} times.")

    # a_num_speakers should be small integer
    def print_total_speaker_times(self,a_num_speakers=2):
        if self.modified_filename : audio_sr = sr.AudioFile(self.modified_filename)
        else : 
            print("\nerror : must use merge_diarization() before printing speaker times!")
            return
        if self.spkr_counts == None: 
            print("\nerror : must use main_word_count() before printing speaker times!")
            return

        self.speaker_times = [0 for i in range(a_num_speakers)]
        for frag in self.merged_fragments:
            for i in range(a_num_speakers):
                if frag[0]==f"SPEAKER_0{i}":
                    self.speaker_times[i] += frag[2]-frag[1]
        print("\n\n##### Speaker Total Times #####")
        for i in range(a_num_speakers):
            print(f"\n>>>> Speaker {i} speaks for {round(self.speaker_times[i],1)} seconds.")

    # verbose set to True means prints things, verbose to False is a quiet run
    def run(self):

        # prepare the audio file by cutting - if required
        if self.do_file_cutting:
            self.cut_audio_file(self.filename)
            audio_file = self.modified_filename
        else :
            audio_file = self.filename

        # do diarization
        self.do_diarization(audio_file)

        #do speech to text on the entire file
        if self.verbose : self.do_and_print_speech_to_text(audio_file,True)

        if self.verbose : self.print_diarization()

        self.merge_diarization()

        if self.verbose : self.print_diarization()

        self.generate_transcript(audio_file)

        if self.verbose : self.print_transcript()


    # deletes the diarization object and modified wav file
    # should only use this after run() and main_word_count()
    def clean(self):
        os.remove(self.modified_filename)
        del self.dia

#TODO! try outputting plots if needed. who said what or who said how much. query frequency of word for each speaker.