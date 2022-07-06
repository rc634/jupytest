# import required libraries
from pydub import AudioSegment 
from pydub.playback import play
from pydub.utils import mediainfo


wav_file = AudioSegment.from_file(file = "audio.wav", format = "wav")

info = mediainfo("audio.wav")


ratio = float(len(wav_file))/float(info["duration_ts"])
sample_rate = wav_file.frame_rate*ratio
start_time = 8.*60.
end_time = 9.*60.
start_index = int(start_time*sample_rate)
end_index = int(end_time*sample_rate)


modified_wav_file = wav_file[start_index:end_index]
modified_wav_file.export(out_f = "segment.wav" , format = "wav")


# Play the audio file
play(modified_wav_file)
