# Project repository for the course FYS5429: Advanced machine learning and data analysis for the physical sciences

### Project description and scope
In this project we are attempting to create a spiking neural network from scratch, closely following these articles/pages for inspiration, and guidance:
- https://www.sciencedirect.com/science/article/abs/pii/S0925231214005785?fbclid=IwAR2Swcsbs7Cpqk8M-o3rNbB9wtmPkZ5fRzoLf41vrx-duKTj4C1iqvUPTAw
- https://github.com/Shikhargupta/Spiking-Neural-Network
- https://medium.com/@tapwi93/first-steps-in-spiking-neural-networks-da3c82f538ad
- https://analyticsindiamag.com/a-tutorial-on-spiking-neural-networks-for-beginners/

We will start by working on making the encoding function properly - setting up the neurons in the input layer, emulating how the neurons connected to the iris function when "looking" at an object. The SNN we are to create will find its labour in classification tasks, and if possible, may be re-structured to perform other tasks as well.
One major obstacle with such a SNN is to properly optimize, or train, the network. This is not straightforward, as the SNN does not have a structure one can use back-propagation to optimize. Meaning, that the SNN is discontinuous in its layers - and we need to find clever ways to train the network without backpropogation (which is generally the way to go for other such feed-forward neural networks).

This is the current scope of the project, and eventually, if we manage to train our network (and automatize this training process) - we will move on to handle more complex problems.
