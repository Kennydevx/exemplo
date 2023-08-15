import random

class GeneticAlgorithm:
    def __init__(self, population_size, code_length):
        self.population_size = population_size
        self.code_length = code_length
        self.population = [self.generate_random_code() for _ in range(population_size)]

    def generate_random_code(self):
        return "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(self.code_length))

    def evaluate_fitness(self, code):
        # Simule a avaliação da qualidade do código
        return sum(1 for char in code if char == "a")

    def evolve_population(self):
        # ... (resto do seu código)

# Exemplo de uso
if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=20, code_length=20)

    # Loop de geração evolutiva simulada
    for generation in range(100):
        ga.evaluate_population_fitness()
        print(f"Generation {generation + 1}: Best code = {ga.get_best_code()}")

        ga.evolve_population()

    print("Final best code:", ga.get_best_code())
