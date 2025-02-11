import numpy as np
import matplotlib.pyplot as plt


# ф-я Розенброка
def rosenbrock(x, y):
    return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)


# начальная популяция
def create_population(size, bounds):
    return [np.random.uniform(bounds[0], bounds[1], 2) for _ in range(size)]


# оценка приспособленности каждого индивида
def evaluate_population(population):
    return [rosenbrock(individual[0], individual[1]) for individual in population]


# селекция (выбор лучших)
def select(population, fitnesses, num_parents):
    sorted_indices = np.argsort(fitnesses)[::-1]
    selected = [population[i] for i in sorted_indices[:num_parents]]
    return selected


# кроссовер (скрещивание)
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = parents[np.random.randint(0, len(parents))]
        parent2 = parents[np.random.randint(0, len(parents))]
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
    return offspring


# мутация
def mutate(offspring, mutation_rate, bounds):
    mutated_offspring = []
    for individual in offspring:
        if np.random.rand() < mutation_rate:
            mutation_index = np.random.randint(0, len(individual))
            individual[mutation_index] += np.random.uniform(-0.5, 0.5)
            individual[mutation_index] = np.clip(individual[mutation_index], bounds[0], bounds[1])
        mutated_offspring.append(individual)
    return mutated_offspring


# параметры
population_size = 100
num_generations = 100
num_parents = 20
mutation_rate = 0.1
bounds = [-2.0, 2.0]

# начальная популяция
population = create_population(population_size, bounds)

# эволюция
best_fitnesses = []
for generation in range(num_generations):
    fitnesses = evaluate_population(population)
    best_fitness = max(fitnesses)
    best_fitnesses.append(best_fitness)

    parents = select(population, fitnesses, num_parents)
    offspring = crossover(parents, population_size - num_parents)
    offspring = mutate(offspring, mutation_rate, bounds)

    population = parents + offspring

    print(f"Generation {generation}: Best Fitness = {best_fitness}")

# нахождение лучшего решения
fitnesses = evaluate_population(population)
best_individual = population[np.argmax(fitnesses)]
print(f"Best solution: x = {best_individual[0]}, y = {best_individual[1]}, f(x, y) = {max(fitnesses)}")

# график
plt.plot(best_fitnesses)
plt.title("Значение фитнес-функции на каждом поколении")
plt.xlabel("Поколение")
plt.ylabel("Фитнес-значение")
plt.show()