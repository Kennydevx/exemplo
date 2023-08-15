import os
import h5py

# Exemplos de códigos de linguagens de programação
example_codes = [
    """print("Hello, World!")""",
    """def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n-1)""",
    """# Calcula a sequência de Fibonacci
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib"""
    # Adicione mais exemplos de código aqui
]

# Caminho absoluto para o arquivo HDF5
file_path = './code_storage.h5'

# Verifica se o arquivo já existe ou não
if os.path.exists(file_path):
    # Abre o arquivo existente
    with h5py.File(file_path, 'a') as f:
        # Remove o dataset 'codes' se já existir
        if 'codes' in f:
            del f['codes']
        # Cria um novo dataset 'codes'
        f.create_dataset('codes', data=[code.encode('utf-8') for code in example_codes])
else:
    # Cria um novo arquivo HDF5
    with h5py.File(file_path, 'w') as f:
        # Cria um dataset 'codes'
        f.create_dataset('codes', data=[code.encode('utf-8') for code in example_codes])
