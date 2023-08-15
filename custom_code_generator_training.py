import os
import h5py
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def load_codes(file_path='code_storage.h5'):
    with h5py.File(file_path, 'r') as f:
        codes = [code.decode('utf-8') for code in f['codes']]
    return codes

def encode_result(code, char_to_index):
    encoded_result = [char_to_index[char] for char in code]
    return encoded_result

def train_model(training_input, training_output, validation_input, validation_output, num_tokens, max_sequence_length, num_classes, char_to_index):
    longest_sequence = max(len(sequence) for sequence in training_input + validation_input)
    training_input_np = np.zeros((len(training_input), longest_sequence))
    validation_input_np = np.zeros((len(validation_input), longest_sequence))

    for i, sequence in enumerate(training_input):
        sequence_encoded = encode_result(sequence, char_to_index)
        training_input_np[i, :len(sequence_encoded)] = sequence_encoded

    for i, sequence in enumerate(validation_input):
        sequence_encoded = encode_result(sequence, char_to_index)
        validation_input_np[i, :len(sequence_encoded)] = sequence_encoded

    model = Sequential([
        Embedding(input_dim=num_tokens, output_dim=32, input_length=longest_sequence),
        LSTM(num_classes, return_sequences=True),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(training_input_np, training_output, validation_data=(validation_input_np, validation_output), epochs=100, batch_size=128)

    evaluation_score = model.evaluate(validation_input_np, validation_output, verbose=0)
    print(f"Avaliação do modelo: Loss = {evaluation_score[0]}, Accuracy = {evaluation_score[1]}")

    model.save('custom_code_generator_model.h5')

def convert_list_to_numpy_array(list):
  """Converte uma lista de listas para um array numpy de uma dimensão"""
  numpy_array = []
  for sublist in list:
    for char in sublist:
      if not char.isalnum():
        numpy_array.append(np.nan)
      else:
        numpy_array.append(np.array(char, dtype=np.int32))
  return np.array(numpy_array)

def main():
    codes = load_codes()
    num_examples = len(codes)

    # Separe os códigos em treinamento e validação
    training_codes = codes[:int(0.8 * num_examples)]
    validation_codes = codes[int(0.8 * num_examples):]

    num_tokens = 128
    max_sequence_length = 100

    # Crie um dicionário para mapear caracteres para índices
    unique_chars = sorted(set(''.join(codes)))
    char_to_index = {char: index for index, char in enumerate(unique_chars)}

    # Converta os códigos para uma lista de sequências de caracteres
    training_input = [list(code) for code in training_codes]
    validation_input = [list(code) for code in validation_codes]

    # Converta as sequências de caracteres para matrizes numéricas
    training_output = [tf.keras.utils.to_categorical(encode_result(code, char_to_index), num_classes=len(unique_chars)) for code in training_input]
    validation_output = [tf.keras.utils.to_categorical(encode_result(code, char_to_index), num_classes=len(unique_chars)) for code in validation_input]

    num_classes = len(unique_chars)  # O número de classes é igual ao número de caracteres únicos

    # Treine o modelo
    train_model(training_input, training_output, validation_input, validation_output, num_tokens, max_sequence_length, num_classes, char_to_index)

    # Gere um código aleatório
    generated_code = generate_random_code(model, char_to_index, num_tokens, max_sequence_length)
    print("Generated Code:")
    print(generated_code)

    # Modifique um código aleatoriamente
    modified_code = modify_randomly(codes[0])
    print("Modified Code:")
    print(modified_code)

if __name__ == "__main__":
    main()
