import h5py

def add_code_to_h5(code):
    current_length = 0  # Inicializa a variável fora do bloco if/else
    with h5py.File('code_storage.h5', 'a') as f:
        if 'codes' not in f:
            codes = f.create_dataset('codes', dtype=h5py.string_dtype(), shape=(1,), maxshape=(None,))
            codes[0] = code
        else:
            codes = f['codes']
            current_length = codes.shape[0]
            codes.resize((current_length + 1,))
            codes[current_length] = code
            codes.flush()
    print(f"Total codes saved: {current_length + 1}")

if __name__ == "__main__":
    new_code = """
import random
import string

length = 12
password = ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(length))
print(f"A senha gerada é: {password}")

"""
    add_code_to_h5(new_code)
