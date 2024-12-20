AI-Driven Dynamic Music Synthesizer with Generative Sound Creation and Real-Time Manipulation

I am interested to create an AI-based music synthesizer that not only generates new sounds from a 
trained model but also allows for real-time manipulation of these sounds using both text and manual 
inputs. This idea combines the fields of Generative Machine Learning, Natural Language Processing 
(NLP), and Audio Signal Processing, making sound design intuitive and accessible for musicians and 
sound engineers.

Steps involved: 
1. Sound Generation: Train a generative model to learn from a diverse database of sounds and 
generate new audio samples on command. 
2. Text-to-Sound Mapping: Implement an NLP system that interprets user descriptions and 
maps them to sound synthesis parameters. 
3. Sound Manipulation: Allow real-time sound modifications using manual controls (like knobs 
for echo and reverb) and further refine these adjustments based on additional text 
commands. 
4. Keyboard Integration: Enable users to play the generated and manipulated sounds in real 
time using a MIDI keyboard or a virtual keyboard interface. 

Directions to run code: 

1. Use to src/notebooks/Ml_Audio.ipynb for feature Extraction
2. Then use the files in the src/data to get audio data
3. Run the src/notebooks/Wavenet_model.ipynb for training and testing model
4. This will take at least half an hour to train and test
3. Run ui.py for the front end
4. This will take an hour to complete generation of an audio file.

