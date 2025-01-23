from src import models
import torch
import audiofile
from IPython.display import Audio, display
import time
import subprocess


class MakeAudio:
    def __init__(self,file_list):
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = "cpu"

        self.demucs = models.Demucs(name="hdemucs_mmi", other_metadata={"segment":2, "split":True},
                        device=self.device, logger=None)
        

        self.count = 0 
        self.audio_file = file_list
        
    
    def wav_audio_from_video(self,video_path):

        "Convert video to wav files"
        output_file = f"sample{self.count}.wav"
        self.audio_file.append(output_file)
        command = [
            "ffmpeg",
            "-i",video_path,
            "-q:a 0 -map a", #For Best quality
            output_file

        ]
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)



    def classify_audio_make_sep_file(self,audio_file,list_of_class_need=["drums"]):

        res = self.demucs(audio_file) #Checked for audio_file
        for cls in res:
            print(f"Classified Classes : {cls}")
            if cls in list_of_class_need:
                part = res[cls] 
                audiofile.write(f"cls_{cls}.mp3", part, self.demucs.sample_rate)




make = MakeAudio()
make.wav_audio_from_video('caught_behind/b_383_2_8_4_0_c1.mp4')
make.classify_audio_make_sep_file('/Users/sirishapeyyala/Downloads/vocals_removal/ultimatevocalremover_api/sample0.wav')





