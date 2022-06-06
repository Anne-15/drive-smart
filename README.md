# Drive Smart Project
A model built to predict the expected time travel of a delivery system.
The model is built with two different technologies; tensorflow and sktlearn. 

# The tensorflow model
A linear regression problem that tends to solve and predict the expected arrival time in seconds given the distance in kilometers. We used the regression neural networks
algorithm to create a model but first the data sample.csv had to be filtered and reduced to 2 columns only (the data that was required). Then the null values eliminated.

# The sklearn model
The process is basically the similar to the tensorflow model the only difference is that we used a different language, python. 

# Saving the models
After bulding the models from the given algorithms, we saved the models so that we don't have to train the algorithm every time we run the model to save on time and space.
The saved model under the folder new_model from the sklearn model. We used pickle to save the model due to its simplicity and reliability.

With the tensorflow model, we saved checkpoints of the training model during the training process. Then a saved model of the trained model after training the check points.

# The main file for the model
app(1).py - an API to export the machine learning and AI model for external applications.
model.py - tensorflow model
model_s.py - sklearn model
new_model - saved model sklearn
