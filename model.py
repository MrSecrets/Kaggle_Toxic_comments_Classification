import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
import parameters


def classifier(n_classes):
	model = Sequential()
	model.add(Embedding(parameters.max_features,
                    parameters.embedding_dims,
                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                    trainable=False))
	model.add(Dropout(0.2))
	model.add(Conv1D(parameters.filters,
                 parameters.kernel_size,
                 padding='valid',
                 activation='relu'))
	model.add(MaxPooling1D())
	model.add(Conv1D(parameters.filters,
	                 5,
	                 padding='valid',
	                 activation='relu'))
	# we use max pooling:
	model.add(GlobalMaxPooling1D())
	# We add a vanilla hidden layer:
	model.add(Dense(parameters.hidden_dims, activation='relu'))
	model.add(Dropout(0.2))

	# We project onto 6 output layers, and squash it with a sigmoid:
	model.add(Dense(n_classes, activation='sigmoid'))
	model.summary()
	return model