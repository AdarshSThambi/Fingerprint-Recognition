import json
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import normalize

# Define the CNN model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(111, activation='softmax')
])

# Load the train labels and minutiae array
labels_file = open('../Data/labels.json', 'r')
labels = json.load(labels_file)
labels_file.close()

minutiae_file = open('../Data/minutiae_array.json', 'r')
minutiae_array = json.load(minutiae_file)
minutiae_file.close()

minutiae_array = np.reshape(minutiae_array, (660, 388, 388, 1))
labels = [int(x) for x in labels]
labels = np.array(labels)

lookup_table = np.eye(111)
labels = lookup_table[labels]
labels = np.reshape(labels, (660, 111))

# Load the test labels and minutiae array
labels_file = open('../Data/labels_test.json', 'r')
labels_test = json.load(labels_file)
labels_file.close()

minutiae_file = open('../Data/minutiae_array_test.json', 'r')
minutiae_array_test = json.load(minutiae_file)
minutiae_file.close()

minutiae_array_test = np.reshape(minutiae_array_test, (220, 388, 388, 1))
labels_test = [int(x) for x in labels_test]
labels_test = np.array(labels_test)

lookup_table = np.eye(111)
labels_test = lookup_table[labels_test]
labels_test = np.reshape(labels_test, (220, 111))

# Compile the CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
model.fit(minutiae_array, labels, epochs=100)

# Evaluate the CNN model
loss, accuracy = model.evaluate(minutiae_array_test, labels_test)
print('Accuracy: {}\nLoss: {}'.format(accuracy, loss))

# Save the model
model.save('CNN_fingerprint.h5')
