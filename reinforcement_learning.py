import numpy as np

class ReinforcementLearningAgent:
    def __init__(self):
        self.q_values = {}

    def choose_action(self, state):
        if state not in self.q_values:
            self.q_values[state] = np.zeros(3)  # 3 ações possíveis

        # Modificação: Adicione exploração inicial (ε-greedy)
        epsilon = 0.1  # Probabilidade de escolher uma ação aleatória
        if np.random.rand() < epsilon:
            return np.random.choice(len(self.q_values[state]))
        else:
            return np.argmax(self.q_values[state])
        
    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_values:
            self.q_values[state] = np.zeros(3)
        if next_state not in self.q_values:
            self.q_values[next_state] = np.zeros(3)
        current_q = self.q_values[state][action]
        max_future_q = np.max(self.q_values[next_state])
        new_q = current_q + 0.1 * (reward + 0.9 * max_future_q - current_q)
        self.q_values[state][action] = new_q

        # Modificação: Ajuste a taxa de aprendizado e o fator de desconto
        learning_rate = 0.9  # Ajuste conforme necessário
        discount_factor = 0.95  # Ajuste conforme necessário

        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        self.q_values[state][action] = new_q

# Exemplo de uso
agent = ReinforcementLearningAgent()

state = "state1"
action = agent.choose_action(state)
reward = 1
next_state = "state2"

agent.update_q_value(state, action, reward, next_state)

# Exemplo de uso
if __name__ == "__main__":
    agent = ReinforcementLearningAgent()

    # Loop de treinamento simulado
    for _ in range(1000):
        state = "initial"
        for _ in range(10):
            action = agent.choose_action(state)
            reward = 1 if action == 2 else 0
            next_state = "final" if action == 2 else "mid"
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

    print("Q-values:", agent.q_values)
