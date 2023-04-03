import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Load data
unsw = pd.read_csv("UNSW-NB15_1.csv")
botiot = pd.read_csv("bot-iot_1.csv")
data = pd.concat([unsw, botiot])

# Preprocessing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = to_categorical(y)

# Define the input shape
input_shape = (64, 64, 3)

# Define the number of classes
num_classes = 15

# Define the rank of the low-rank approximation
rank = 20

# Define the input layer
input_layer = Input(shape=input_shape)

# Define the convolutional layers
conv_layer_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(input_layer)
max_pooling_layer_1 = MaxPooling2D(pool_size=(2, 2))(conv_layer_1)
conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(max_pooling_layer_1)
max_pooling_layer_2 = MaxPooling2D(pool_size=(2, 2))(conv_layer_2)

# Flatten the output from the convolutional layers
flatten_layer = Flatten()(max_pooling_layer_2)

# Define the MLP layers
dense_layer_1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(flatten_layer)
dropout_layer = Dropout(rate=0.5)(dense_layer_1)
dense_layer_2 = Dense(units=num_classes, activation='softmax')(dropout_layer)

# Define the low-rank approximation layers
svd_layer_1 = tf.linalg.SVD(rank, full_matrices=False, compute_uv=True)
dense_layer_1_u, dense_layer_1_s, dense_layer_1_v = svd_layer_1(dense_layer_1)

svd_layer_2 = tf.linalg.SVD(rank, full_matrices=False, compute_uv=True)
dense_layer_2_u, dense_layer_2_s, dense_layer_2_v = svd_layer_2(dense_layer_2)

# Combine the low-rank approximation layers with the MLP layers
dense_layer_1 = tf.matmul(dense_layer_1_u, tf.matmul(tf.linalg.diag(dense_layer_1_s), tf.transpose(dense_layer_1_v)))
dense_layer_2 = tf.matmul(dense_layer_2_u, tf.matmul(tf.linalg.diag(dense_layer_2_s), tf.transpose(dense_layer_2_v)))

# Define the model
model = Model(inputs=input_layer, outputs=dense_layer_2)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
