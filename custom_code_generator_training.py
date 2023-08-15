import os
import h5py
import subprocess
import numpy as np
import tensorflow as tf  # Adicione esta linha
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def load_codes(file_path='code_storage.h5'):
  with h5py.File(file_path, 'r') as f:
    codes = [code.decode('utf-8') for code in f['codes']]
  return codes


def save_code_to_file(code, filename):
  with open(filename, 'w') as f:
    f.write(code)


def execute_code_file(filename):
  try:
    result = subprocess.check_output(['python', filename], stderr=subprocess.STDOUT, text=True)
    return result.strip()
  except subprocess.CalledProcessError as e:
    return f"Erro na execução: {e.output.strip()}"


def encode_result(code, char_to_index):
    encoded_result = np.zeros(len(code), dtype=np.int32)
    for i, char in enumerate(code):
        encoded_result[i] = char_to_index[char]
    return encoded_result


def generate_random_code(model, token_to_index, num_tokens, max_sequence_length):
  generated_code = ''
  input_sequence = np.zeros((1, max_sequence_length))

  for i in range(max_sequence_length):
    predictions = model.predict(input_sequence)
    predicted_index = np.argmax(predictions)
    generated_char = next(char for char, index in token_to_index.items() if index == predicted_index)
    if generated_char == '\0':
      break
    generated_code += generated_char
    input_sequence[0, i] = predicted_index

  return generated_code


def modify_randomly(code):
  # Adicione sua lógica de modificação aleatória aqui
  # Por exemplo, pode trocar alguns caracteres, adicionar ou remover linhas, etc.
  return code


def train_model(training_input, training_output, validation_input, validation_output, num_tokens, max_sequence_length):
    model = Sequential([
        Embedding(input_dim=num_tokens, output_dim=32, input_length=max_sequence_length),
        LSTM(num_classes, return_sequences=True),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(training_input, training_output, validation_data=(validation_input, validation_output), epochs=100, batch_size=128)


    evaluation_score = model.evaluate(validation_input, validation_output, verbose=0)
    print(f"Avaliação do modelo: Loss = {evaluation_score[0]}, Accuracy = {evaluation_score[1]}")

    model.save('custom_code_generator_model.h5')

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
    training_output = [encode_result(code, char_to_index) for code in training_input]
    validation_output = [encode_result(code, char_to_index) for code in validation_input]

    num_classes = len(char_to_index)  # O número de classes é igual ao número de caracteres únicos

    # Treine o modelo
    train_model(training_input, training_output, validation_input, validation_output, num_tokens, max_sequence_length, num_classes)

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
