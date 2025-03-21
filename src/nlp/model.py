import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.words = None
        self.classes = None
        self.tokenizer = None
    
    def build_model(self, input_shape, output_shape):
        # Creating the model - 3 layers
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_shape, activation='softmax'))
        
        # Compile model
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        return self.model
    
    def train(self, train_x, train_y, epochs=200, batch_size=5):
        # Fit the model
        hist = self.model.fit(
            np.array(train_x), np.array(train_y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save the model
        self.model.save('models/intent_model.h5')
        return hist
    
    def load_model(self):
        self.model = tf.keras.models.load_model('models/intent_model.h5')
        self.words = pickle.load(open('models/words.pkl', 'rb'))
        self.classes = pickle.load(open('models/classes.pkl', 'rb'))
        self.tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
        
    def predict_class(self, sentence, preprocessor, error_threshold=0.25):
        # Generate probabilities from the model
        input_data = preprocessor.prepare_input(sentence)
        results = self.model.predict(input_data)[0]
        
        # Filter out predictions below threshold
        results = [[i, r] for i, r in enumerate(results) if r > error_threshold]
        
        # Sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
            
        return return_list