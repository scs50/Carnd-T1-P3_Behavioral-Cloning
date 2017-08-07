from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import csv

image_paths = []
angles = []
#Open csv with angles and image locations and read in
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # get center angle and image path
        image_paths.append(line[0])#.split('/')[-1])
        angles.append(float(line[3]))
        # add left correction angle and image
        image_paths.append(line[1])#.split('/')[-1])
        angles.append(float(line[3])+0.25)
        # add right correction angle and image
        image_paths.append(line[2])#.split('/')[-1])
        angles.append(float(line[3])-0.25)

image_paths = np.array(image_paths)
angles = np.array(angles)

print('Before:', image_paths.shape, angles.shape)
# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

print('After:', image_paths.shape, angles.shape)

def add_sw_angle_to_image(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to displaying. 
    Converts colorspace back to original BGR, applies text to display steering angle and 
    frame number (within batch to be visualized), and applies lines representing steering angle 
    and model-predicted steering angle (if available) to image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    # img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(image,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    # apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+pred_angle*w/4),int(h/2)),(0,0,255),thickness=4)
    return img


def generate_data_for_visualization(image_paths, angles, batch_size=5, validation_flag=False):
	"""
	Apply preprocessing to image for visualization
	"""
	X = []
	y = []
	image_paths, angles = shuffle(image_paths, angles)
	for i in range(batch_size):
		img = cv2.imread(image_paths[i])
		angle = angles[i]
		img = img[50:140,:,:]
		img = cv2.resize(img,(200, 66))
		X.append(img)
		y.append(angle)
	return (np.array(X), np.array(y))


def display_image_with_sw_angle(X,y,y_pred=None):
    '''
    format the data from the dataset (image, steering angle) and display wtih cv2
    '''
    for i in range(len(X)):
        if y_pred is not None:
            img = add_sw_angle_to_image(X[i], y[i], y_pred[i], i)
        else:
            img = add_sw_angle_to_image(X[i], y[i], None, i)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# visualize a single batch of the data
X,y = generate_data_for_visualization(image_paths, angles)
display_image_with_sw_angle(X,y)

#split training and validation sets
image_paths_train, image_paths_validation, angles_train, angles_validation = train_test_split(image_paths, angles, test_size=0.1, random_state=14)

print('Train:', image_paths_train.shape, angles_train.shape)
print('Validation:', image_paths_validation.shape, angles_validation.shape)


def process_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    image = image[50:140,:,:]
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200, 66))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


def generator(image_paths, angles, batch_size=64, validation_flag=False):

    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])

    num_samples = len(angles)
    while True: # Loop forever so the generator never terminates
        for i in range(num_samples):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = process_image(img)
# 
            X.append(img)
            y.append(angle)

            #Double/even out data by flipping image and steering angle
            img = cv2.flip(img, 1)
            angle *= -1
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)


# compile and train the model using the generator function
train_generator = generator(image_paths_train, angles_train, batch_size=64, validation_flag=False)

validation_generator = generator(image_paths_validation, angles_validation, batch_size=64, validation_flag=True)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(66,200,3)))

# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# model.add(Dropout(0.50))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(1))

model.compile(optimizer=Adam(lr=1e-4), loss='mse')
# model.fit_generator(train_generator, samples_per_epoch =   len(train_samples), validation_data=validation_generator,      nb_val_samples=len(validation_samples), nb_epoch=3)


#Visualize Loss
history_object = model.fit_generator(train_generator, validation_data =
    validation_generator, nb_epoch=15, verbose=1,
    nb_val_samples=2560, samples_per_epoch=24000)
     

print(model.summary())

### print the keys contained in the history object
print(history_object.history.keys())

X,y = generate_data_for_visualization(image_paths, angles)
display_image_with_sw_angle(X,y)

# visualize some predictions
n=10
X_test,y_test = generate_data_for_visualization(image_paths[:n], angles[:n], batch_size = n, validation_flag=True)
y_pred = model.predict(X_test, n, verbose=2)
display_image_with_sw_angle(X_test, y_test, y_pred)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
