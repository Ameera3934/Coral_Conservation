Emotion Based, Text-to-Music, Music Generator to Enhance Engagement in Coral Reef Degradation Education

## Setup
    - conda create --name emotions python=3.10
    - conda activate emotions
    - cd Ameera_Shahid
    - pip install -r requirements.txt
    - python __main__.py

## Walkthrough
    - After running python __main__.py, a window should open which is the app. This page will contain 5 paragraphs of text, and at the bottom of each text there is a corresponding generate audio button. 
    - Click the first generate audio button before reading the text
    - Once the music starts playing, read the first paragraph - the music will play for 30 seconds 
        - Currently, if the emotion is already saved as a file name with an attached wav file, it wont generate a new song - it will play the corresponding song for the emotion that is predicted. If there are no wav files, the music generator will take a while to generate a new 30 second song with good quality (higher number of interference steps). So, it may take a couple of minutes (which is why I've done it so that it saves the file and runs the corresponding emotion song when that emotion is predicted - otherwise it generates a new song everytime and the usability decreases)
    - The emotion of the text will be displayed in the terminal you're running the app from - this is predicted from the emotion classifier in the data_model.py file
    - Go through each paragraph to learn about coral reef degradation and what you can do

## Disclaimer
    - For this project, i utilised the Emotions dataset created by Nidula Elgiriyewithana and shared on Kaggle. This dataset contains approximately 417,000 example sentences, each labelled with one of 6 emotions: sadness, joy, anger, love, fear, and surprise. This dataset was chosen as it provides a range of emotions that range from strongly positive (joy) to strongly negative (anger). The dataset has 3 columns: column 1 is the numerical identifier; column 2 is text; column 3 is the emotion label. 
        - This dataset is saved as text.csv 
        - Once i pre-processed the dataset for the logistic regression model, i saved the updated data as simplified.csv to allow a shorter run time and keep everything organised
    
    - for this project, i also utilised the musiLDM music genertaor from hugging face. The generator takes in a text prompt – which in the case of this project, the prompt will be an emotion – and that will trigger the generation process. The prompts need to be as descriptive as possible to generate the best output.
    
        - License:
            Copyright 2024 The HuggingFace Team. All rights reserved

            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.

## References
    - Elgiriyewithana, N. (2024) Emotions, Kaggle. Available at: https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data (Accessed: 23 April 2024). 
    - von Platen, P. et al. (2022) Huggingface/Diffusers: diffusers: State-of-the-art diffusion models for image and audio generation in pytorch and flax., GitHub. Available at: https://github.com/huggingface/diffusers (Accessed: 25 April 2024).  repository-code: 'https://github.com/huggingface/diffusers' abstract: >- Diffusers provides pretrained diffusion models across multiple modalities, such as vision and audio, and serves as a modular toolbox for inference and training of diffusion models. keywords: - deep-learning - pytorch - image-generation - hacktoberfest - diffusion - text2image - image2image - score-based-generative-modeling - stable-diffusion - stable-diffusion-diffusers license: Apache-2.0 version: 0.12.1
