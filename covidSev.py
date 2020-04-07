from numpy import loadtxt
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load the dataset
dataset = loadtxt('covid.csv', 
delimiter=',')
X = dataset[:,0:5] 
y = dataset[:,5]

# define the keras model
model = Sequential() 
model.add(Dense(12, input_dim=5, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1, activation='relu'))

# compile the keras model
model.compile(loss='binary_crossentropy', 
optimizer='adam', 
metrics=['accuracy'])

# save trained model

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# fit the keras model on the 
# dataset
model.fit(X, y, epochs=5, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# predict a 60+ year old
# test = np.array([0,0,0,1])
# model.predict(test)
