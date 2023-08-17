import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, code_length, mutation_rate):
        self.population_size = population_size
        self.code_length = code_length
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            code = np.random.randint(0, 5, size=self.code_length)  # Replace 3 with the number of available actions
            population.append(code)
        return population
    
    def evaluate_population_fitness(self, fitness_function):
        fitness_scores = []
        for code in self.population:
            fitness_score = fitness_function(code)
            fitness_scores.append(fitness_score)
        return fitness_scores
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.code_length)  # Choose a random crossover point
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child
    
    def mutate(self, code):
        mutated_code = code.copy()
        for i in range(self.code_length):
            if np.random.rand() < self.mutation_rate:
                mutated_code[i] = np.random.randint(0, 5)  # Replace 3 with the number of available actions
        return mutated_code
    
    def evolve_population(self, fitness_scores):
        new_population = []
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(fitness_scores)
            parent2 = self.select_parent(fitness_scores)
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        
    
    def select_parent(self, fitness_scores):
        total_fitness = np.sum(fitness_scores)
        if total_fitness <= 0:
            parent_index = np.random.choice(range(len(self.population)))
        else:
            probabilities = fitness_scores / total_fitness
            parent_index = np.random.choice(range(len(self.population)), p=probabilities)
        return self.population[parent_index]
    
    def select_parent_tournament(self, fitness_scores, tournament_size=3):
        tournament_indices = np.random.choice(len(self.population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        parent_index = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[parent_index]
    def crossover_two_point(self, parent1, parent2):
        crossover_points = np.sort(np.random.choice(self.code_length, size=2, replace=False))
        child = np.concatenate((parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]], parent1[crossover_points[1]:]))
        return child
    
    def get_best_code(self, fitness_scores):
        best_index = np.argmax(fitness_scores)
        return self.population[best_index]

# Example usage
def fitness_function(target_code, code):
    if np.all(np.array(code) == np.array(target_code)):
        score = len(target_code)  # Increase the score when the code matches the target
    else:
        score = 0
    return score

def code_similarity(target_code, generated_code):
    similarity_score = np.sum(np.array(target_code) == np.array(generated_code))
    return similarity_score

def code_similarity_advanced(target_code, generated_code):
    target_lines = target_code.split('\n')
    generated_lines = generated_code.split('\n')
    
    matcher = difflib.SequenceMatcher(None, target_lines, generated_lines)
    similarity_score = matcher.ratio()  # Calcula a similaridade baseada nas sequências
    
    return similarity_score

if __name__ == "__main__":
    population_size = 2000
    code_length = 20
    mutation_rate = 0.2

    target_code = ["print('Hello')", "for i in range(5):", "x = 2 + 3", "if x > 5:", "x -= 1", "while x > 0:"]
    
    ga = GeneticAlgorithm(population_size, code_length, mutation_rate)

    for generation in range(10):
        generated_codes = ga.population
        
        # Avaliar a similaridade entre os códigos gerados e o código alvo
        fitness_scores = [code_similarity_advanced('\n'.join(target_code), '\n'.join(code)) for code in generated_codes]
        
        best_code_index = np.argmax(fitness_scores)
        best_code = generated_codes[best_code_index]

        print(f"Generation {generation + 1}: Best code = {best_code}")

        ga.evolve_population(fitness_scores)
