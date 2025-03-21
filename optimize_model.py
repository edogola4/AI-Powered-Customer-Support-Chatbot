import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import time
import matplotlib.pyplot as plt
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.model import IntentClassifier
import json

def optimize_model():
    """Optimize the model and evaluate its performance"""
    print("Loading model and data...")
    
    # Load the model and preprocessor
    preprocessor = TextPreprocessor('data/intents/intents.json')
    words, classes, documents = preprocessor.preprocess()
    train_x, train_y = preprocessor.create_training_data()
    
    # Initialize classifier
    classifier = IntentClassifier()
    
    # Test different model configurations
    layer_configs = [
        [(128, 0.5), (64, 0.5)],              # Original
        [(256, 0.5), (128, 0.5)],             # Larger
        [(128, 0.5), (64, 0.5), (32, 0.5)],   # Deeper
        [(64, 0.3), (32, 0.3)]                # Smaller with less dropout
    ]
    
    results = []
    
    for i, config in enumerate(layer_configs):
        print(f"\nTesting configuration {i+1}:")
        for layer in config:
            print(f"- Dense({layer[0]}) with Dropout({layer[1]})")
        
        # Build model with this configuration
        model = build_custom_model(len(words), len(classes), config)
        
        # Compile model
        sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        # Train model
        start_time = time.time()
        history = model.fit(
            np.array(train_x), np.array(train_y),
            epochs=100,
            batch_size=5,
            verbose=0
        )
        end_time = time.time()
        
        # Evaluate model
        loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y), verbose=0)
        training_time = end_time - start_time
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Get model size
        model.save('models/temp_model.h5')
        model_size = os.path.getsize('models/temp_model.h5') / (1024 * 1024)  # in MB
        print(f"Model size: {model_size:.2f} MB")
        
        results.append({
            'config': config,
            'accuracy': accuracy,
            'loss': loss,
            'training_time': training_time,
            'model_size': model_size,
            'history': history.history
        })
    
    # Find the best model
    best_model_idx = np.argmax([r['accuracy'] for r in results])
    best_model = results[best_model_idx]
    
    print("\n=== Best Model Configuration ===")
    for layer in best_model['config']:
        print(f"- Dense({layer[0]}) with Dropout({layer[1]})")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Loss: {best_model['loss']:.4f}")
    print(f"Training time: {best_model['training_time']:.2f} seconds")
    print(f"Model size: {best_model['model_size']:.2f} MB")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    accuracies = [r['accuracy'] for r in results]
    plt.bar(range(len(results)), accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xticks(range(len(results)), [f"Config {i+1}" for i in range(len(results))])
    plt.ylim(0.8, 1.0)
    
    # Training time comparison
    plt.subplot(2, 2, 2)
    times = [r['training_time'] for r in results]
    plt.bar(range(len(results)), times)
    plt.title('Training Time Comparison')
    plt.xticks(range(len(results)), [f"Config {i+1}" for i in range(len(results))])
    
    # Model size comparison
    plt.subplot(2, 2, 3)
    sizes = [r['model_size'] for r in results]
    plt.bar(range(len(results)), sizes)
    plt.title('Model Size Comparison')
    plt.xticks(range(len(results)), [f"Config {i+1}" for i in range(len(results))])
    
    # Learning curve of best model
    plt.subplot(2, 2, 4)
    plt.plot(best_model['history']['accuracy'])
    plt.plot(best_model['history']['loss'])
    plt.title('Best Model Learning Curve')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('model_optimization_results.png')
    plt.close()
    
    # Save the best model configuration
    with open('models/best_config.json', 'w') as f:
        json.dump({
            'config': [[int(layer[0]), float(layer[1])] for layer in best_model['config']],
            'accuracy': float(best_model['accuracy']),
            'loss': float(best_model['loss'])
        }, f)
    
    print("Optimization complete. Results saved to model_optimization_results.png")

def build_custom_model(input_shape, output_shape, layer_config):
    """Build a model with custom layer configuration"""
    model = tf.keras.models.Sequential()
    
    # Add first layer with input shape
    model.add(tf.keras.layers.Dense(layer_config[0][0], input_shape=(input_shape,), activation='relu'))
    model.add(tf.keras.layers.Dropout(layer_config[0][1]))
    
    # Add additional layers
    for neurons, dropout_rate in layer_config[1:]:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Add output layer
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    
    return model

if __name__ == "__main__":
    import os
    optimize_model()