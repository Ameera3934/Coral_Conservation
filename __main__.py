import os
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Scrollbar, Canvas
import pygame
import scipy
import torch
from diffusers import MusicLDMPipeline
from data_model import predict_emotion

#initialize the device - CPU or GPU
#if torch.cuda.is_available():
   # device = "cuda"
#else:
device = "cpu"

#call on the musicLDM music genertor pipeline - pretrained diffuser model
#bring to device
repo_id = "ucsd-reach/musicldm"
pipeline = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
pipeline = pipeline.to(device)

#generate music function that takes in the emotion predicted
def generate_music_from_emotion(emotion):

    #predetermined number of inference steps - best audio quality
    num_inference_steps = 200

    #audio length set to 30 seconds
    audio_length_in_s = 30

    #use the emotion phrase as the file name
    file_name = uniform_file_name(emotion)

    #save file in audio_file folder
    file_path = f"audio_files/{file_name}.wav"

    #check if that emotion already has a song (path exists) - if so return that songs path
    if os.path.exists(file_path):
        return file_path

    #generate music based on the provided emotion if the emotion doesnt currently have a song
    else:
        #generate new music by calling the pipleine and initalising the main variables
        data = pipeline(
            prompt=emotion,
            audio_length_in_s=audio_length_in_s,
            num_inference_steps=num_inference_steps
        ).audios[0]
        #save generated music as a wav file using scipy
        scipy.io.wavfile.write(file_path, rate=16000, data=data)

        #return path to wav file
        return file_path

#change file name to replace any spaces or sentence enders with a underscore to make it uniform and easily accessable
def uniform_file_name(file_name: str) -> str:
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(".", "_")
    file_name = file_name.replace(",", "_")
    return file_name

#function to generate music when button is clicked - takes in label widget which containts text
def generate_audio(label_widget):
    
    #get text from the label
    text = label_widget.cget("text")
    
    #predict emotion from the text by calling on predict_emotion function from data_mode.py
    emotion = predict_emotion(text)
    
    #generate music based on the predicted emotion 
    file_path = generate_music_from_emotion(emotion)
    
    #use pygame to load the wav file form the path and play the file
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

#main window
def main():
    
    #create the main window
    window = tk.Tk()

    #set window size
    window.geometry("800x500")
    
    #set the title of the window
    window.title("Emotion-Based Music Generation") 

    #create the outer frame to contain canvas and scrollbar - fills the whole window
    outer_frame = tk.Frame(window)
    outer_frame.pack(fill=tk.BOTH, expand=1)

    #create the canvas that fills the whole outer frame
    canvas = tk.Canvas(outer_frame, bg="#0077BE", width=800, height=500, highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    #create the scrollbar 
    scrollbar = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    #bind the canvas to the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)

    #create the inner frame on the canvas to contain text and buttons
    target_frame = tk.Frame(canvas, bg='#0077BE', width=700)
    canvas.create_window((800 - 700) // 2, 0, window=target_frame, anchor="nw")
    
    #title for main page
    title_label = tk.Label(target_frame, text="Coral Reef Degradation", fg="white", bg="#0077BE", font=("Helvetica", 20, "bold"))
    title_label.pack(pady=10)

    #list of prompts for labels
    prompts = [
       "Coral reefs are vital ecosystems that are teeming with life. They are home to a quarter of marine species (Fisher et al., 2015). They offer sanctuary and sustenance to many life forms, and shield coastlines from erosion. In places like Australia, they're not just a wonder of nature, they’re economic powerhouses, generating billions and fostering livelihoods (Reef Authority, 2023).",
       "However, human actions are slowly destroying these underwater wonders. Sadly, between 2009 and 2020, a significant 14% of the world's coral reefs were lost with no hope of revival (UN, 2021). The primary culprit? Coral bleaching - a consequence of rising water temperatures - stripping corals of their vivid hues and exposing them to the brink of extinction. The toll is deeply disheartening; in just a single year, the Caribbean mourned the loss of over half its corals, a poignant reflection of the gravity of this crisis (US Department of Commerce, 2010).",
       "Our world becomes more industrialised, there is anger amongst some as Greenpeace (2019) revealed big corporations like Coca-Cola, PepsiCo, and Nestlé shoulder significant blame for polluting our oceans with plastic. Their relentless pursuit of profit shows a blatant disregard for nature's balance, leaving behind destruction and devastation. What's truly infuriating is their indifference to the environmental havoc they wreak. Despite overwhelming evidence and urgent calls for action, these corporations continue to prioritise profits over the planet's well-being, which should anger many to witness such disregard towards the imminent danger they're creating.",
       "Due to the alarming trends highlighted in 'The World Counts' (2024), scientists predict that by 2050, all corals will be threatened with extinction, with 75 percent facing high to critical threat levels. The thought of a world without thriving coral reefs terrifies many, leaving them scared for the fate of future generations who may never get to witness the reefs and their ecological importance. But what is truly frightening is the widespread lack of care and action in the face of this looming crisis. It's terrifying to witness the total indifference towards safeguarding such a vital ecosystem, knowing that time is running out.",
       "However, despite some’s lack of care, it's heartening to see the efforts being made to combat coral reef destruction. Organizations worldwide are stepping up, implementing conservation projects, and advocating for change. Individuals too can play a crucial role by adopting eco-friendly practices at home, such as reducing single-use plastics and supporting sustainable seafood choices. Conserving water and reducing energy consumption can also contribute positively. Together, we can still make a difference and ensure a brighter future for our coral reefs, bringing joy to future generations who won’t have to miss out on these wonders.",
       
    ]




    #create and place labels and buttons in the middle of the page with one text followed by button
    for prompt in prompts:
        label = tk.Label(target_frame, text=prompt, fg="white", bg="#0077BE", wraplength=700)
        label.pack(pady=5, anchor="center")  # Align textboxes in the middle
        generate_button = tk.Button(target_frame, text="Generate Audio", command=lambda label=label: generate_audio(label))
        generate_button.pack(pady=5, anchor="center")  # Align buttons in the middle

    #create the label for references
    references_label = tk.Label(target_frame, text="References", fg="white", bg="#0077BE", font=("Helvetica", 16, "bold"))
    references_label.pack(pady=10)

    #set the references text
    references = """ - Fisher, R., O'Leary, R. A., Low-Choy, S., Mengersen, K., Knowlton, N., Brainard, R. E., et al. (2015). Species richness on coral reefs and the pursuit of convergent global estimates. Curr. Biol. 25, 500–505. doi: 10.1016/j.cub.2014.12.022
    - Reef Authority (2022) Reef facts, Great Barrier Reef Marine Park Authority. Available at: https://www2.gbrmpa.gov.au/learn/reef-facts#:~:text=The%20Great%20Barrier%20Reef%20is,around%2064%2C000%20full%2Dtime%20jobs. (Accessed: 26 November 2023)
    - UN environment programme (2021) Rising sea surface temperatures driving the loss of 14 percent of corals since 2009, UN Environment programme. Available at: https://www.unep.org/news-and-stories/press-release/rising-sea-surface-temperatures-driving-loss-14-percent-corals-2009#:~:text=release%20Nature%20Action-,Rising%20sea%20surface%20temperatures%20driving%20the%20loss,percent%20of%20corals%20since%202009&text=Nairobi%2C%205%20October%202021%20%2D%20The,the%20world%27s%20coral%20since%202009. (Accessed: 26 November 2023).
    - US Department of Commerce, N.O. and A.A. (2010) What is coral bleaching?, NOAA’s National Ocean Service. Available at: https://oceanservice.noaa.gov/facts/coral_bleach.html (Accessed: 26 November 2023)
    - Greenpeace (2019) Coke, Nestlé, and Pepsico are top plastic polluters yet again, Greenpeace UK. Available at: https://www.greenpeace.org.uk/news/coke-nestle-and-pepsico-are-top-plastic-polluters-yet-again/#:~:text=Volunteers%20all%20over%20the%20world,on%20recycling%20rather%20than%20reduction. (Accessed: 28 April 2024)
    ."""

    #add the reference text to the references label
    references = tk.Label(target_frame, text=references, justify="left", bg="#001F3F", fg="white", wraplength=700)
    references.pack(pady=5, anchor="center")
    
    # Bind the event for adjusting the scroll region with a lambda function - otherwise it continues to scroll down
    references.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

    window.mainloop()


    
#entry point
if __name__ == "__main__":
    main()


