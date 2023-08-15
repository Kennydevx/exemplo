import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from reinforcement_learning import ReinforcementLearningAgent
from genetic_algorithm import GeneticAlgorithm

def load_codes(file_path='code_storage.txt'):
    with open(file_path, 'r') as f:
        codes = f.read().splitlines()
    return codes

def load_additional_data(file_path='additional_code_storage.txt'):
    with open(file_path, 'r') as f:
        codes = f.read().splitlines()
    return codes

def encode_sequence(sequence, char_to_index):
    encoded_sequence = [char_to_index[char] for char in sequence]
    return encoded_sequence

def generate_model(input_shape, num_classes):
    model = Sequential([
        Embedding(input_dim=num_classes, output_dim=528, input_length=input_shape[1]),
        LSTM(64, return_sequences=True),
        Dropout(0.02),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.001)  # Lower learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def generate_random_code(model, index_to_char, num_classes, max_sequence_length):
    seed = np.random.randint(num_classes, size=(1, max_sequence_length))
    generated_sequence = model.predict(seed).argmax(axis=-1)
    generated_code = ''.join(index_to_char[idx] for idx in generated_sequence[0])
    return generated_code

def main():
    codes = load_codes()
    unique_chars = sorted(set(''.join(codes)))
    char_to_index = {char: index for index, char in enumerate(unique_chars)}
    index_to_char = {index: char for char, index in char_to_index.items()}

    num_examples = len(codes)
    training_codes = codes[:int(0.8 * num_examples)]
    validation_codes = codes[int(0.8 * num_examples):]

    max_sequence_length = max(len(code) for code in codes)
    num_classes = len(unique_chars)

    training_input = [encode_sequence(code, char_to_index) for code in training_codes]
    validation_input = [encode_sequence(code, char_to_index) for code in validation_codes]

    training_input = tf.keras.preprocessing.sequence.pad_sequences(training_input, maxlen=max_sequence_length, padding='post', dtype=np.int32)
    validation_input = tf.keras.preprocessing.sequence.pad_sequences(validation_input, maxlen=max_sequence_length, padding='post', dtype=np.int32)

    training_output = tf.keras.utils.to_categorical(training_input, num_classes=num_classes)
    validation_output = tf.keras.utils.to_categorical(validation_input, num_classes=num_classes)

    try:
        model = load_model('trained_model.h5')
        print("Loaded existing model.")
    except:
        model = generate_model(training_input.shape, num_classes)
        print("Generated new model.")

    checkpoint = ModelCheckpoint('trained_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Treinamento inicial com dados originais
    model.fit(training_input, training_output, validation_data=(validation_input, validation_output), epochs=100, batch_size=128, callbacks=[checkpoint])

    # Carregar e preparar novos dados para treinamento
    new_data_input = load_additional_data('additional_code_storage.txt')
    new_data_input_encoded = []  # Lista para armazenar sequências codificadas

    for seq in new_data_input:
        encoded_seq = encode_sequence(seq, char_to_index)
        new_data_input_encoded.append(encoded_seq)

    new_data_input_padded = tf.keras.preprocessing.sequence.pad_sequences(new_data_input_encoded, maxlen=max_sequence_length, padding='post', dtype=np.int32)
    new_data_output_categorical = tf.keras.utils.to_categorical(new_data_input_padded, num_classes=num_classes)

    # Continuar treinamento com novos dados
    model.fit(new_data_input_padded, new_data_output_categorical, epochs=50, batch_size=128, callbacks=[checkpoint])

    generated_code = generate_random_code(model, index_to_char, num_classes, max_sequence_length)
    print("Generated Code:")
    print(generated_code)

    # Usar Reinforcement Learning para gerar código
    rl_agent = ReinforcementLearningAgent()

    for _ in range(10):
        state = "initial"
        generated_code = ""
        for _ in range(max_sequence_length):
            action = rl_agent.choose_action(state)
            generated_code += index_to_char[action]
            next_state = "final" if action == 2 else "mid"
            state = next_state

        print("Generated Code (RL):")
        print(generated_code)

    # Usar Algoritmo Genético para gerar código
    ga = GeneticAlgorithm(population_size=20, code_length=max_sequence_length)

    for generation in range(10):
        ga.evaluate_population_fitness()
        best_code = ga.get_best_code()

        print(f"Generation {generation + 1}: Best code (GA) = {best_code}")

        ga.evolve_population()

    print("Final best code (GA):", ga.get_best_code())

if __name__ == "__main__":
    main()