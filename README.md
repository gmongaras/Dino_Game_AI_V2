# Dino_Game_AI_V2
AI that learns to play the Google dino game using the NEAT algorithm.


The AI is using the genetic algorithm. So, each dino initially makes random moves depending on its inputs into the neural network. Once all dinos die, a new population is created from a weighted selection. The algorithm used in the original repo wasn't really using based on NEAT. The algorithm used for this new repo is based on the original NEAT paper.


Open index.html on a webserver in order to run. Press the up arrow to start the training process.
At the current settings, 500 dinos will be in each population, and each dino has a neural network with 10 inputs and 3 outputs. These hyperparameters can be changed within "script.js". Unlike with Dino_Game_AI (the original repo), the size of the hidden layers are learned and not hardcoded.


The videos within this repo give examples of the early and later parts of the training process. The first video, "Early_Stages_Of_Training.mp4" shows the dinos learning from the first generation to the end of the fifth generation. Since the training process takes a long time, the second video, "Later_Stages_Of_Training.mp4" shows the dinos progress starting at generation 265. The generation number is in the bottom right corner of the screen within the console.
