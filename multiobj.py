from deap import base, creator, tools
from driving_models import *
from utils import *
from keras.preprocessing.image import *
from imageio import imsave
import matplotlib.pyplot as plt

# Define the problem-specific parameters
POPULATION_SIZE = 500
GENERATIONS = 50
ERROR_THRESHOLD = 0.005  # 预测误差阈值

# Image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# Define the input tensor
input_tensor = Input(shape=input_shape)

# Load the model and initialize coverage tables
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model_layer_dict1 = init_coverage_tables1(model1)
layer_name1, index1 = neuron_to_cover(model_layer_dict1)

# Define functions to calculate loss1 and loss1_neuron
loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])

# Define the problem-specific functions
def evaluate(individual):
    # Load the image corresponding to the individual
    img_path = individual
    gen_img = preprocess_image(img_path)
    # Calculate the objectives
    loss1_value = -iterate1([gen_img])[0]
    #print("loss1_value:", loss1_value)
    loss1_neuron_value = -iterate2([gen_img])[0]
    #print("loss1_neuron_value:", loss1_neuron_value)

    # 计算预测误差
    pred_error = loss1_value + ERROR_THRESHOLD  # 期望预测误差为0.01，如果误差大于0.01，则视为不满足约束条件

    # 满足约束条件的个体适应度为实际误差，不满足约束条件的个体适应度设置为一个很大的值
    if pred_error > 0:
        fitness = (100.0, 100.0)  # 设置很大的适应度值，表示不满足约束条件
    else:
        fitness = (loss1_value, loss1_neuron_value)
        update_coverage(gen_img, model1, model_layer_dict1)

    # Return a tuple of objectives
    return fitness

# Create a fitness class with multiple objectives
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

# Create an individual class
creator.create("Individual", str, fitness=creator.FitnessMulti)

# Initialize the toolbox
toolbox = base.Toolbox()

# Define functions for creating individuals and populations
def create_individual():
    return random.choice(img_paths)

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the necessary functions
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)

# Define the mutation function
def mutate_image(individual, mutation_rate=0.05):
    # Load and preprocess the image corresponding to the individual
    gen_img = preprocess_image(individual)

    mutated_individual = np.copy(gen_img)  # 复制图像数据，以避免直接修改原始数据

    # 确定是否对图像进行变异
    if np.random.rand() < mutation_rate:
        # 生成随机变异点的坐标
        num_pixels_to_mutate = int(mutation_rate * gen_img.size)
        mutation_indices = np.random.choice(range(gen_img.size), num_pixels_to_mutate, replace=False)

        # 对选定的像素进行随机修改
        for idx in mutation_indices:
            pixel_idx = np.unravel_index(idx, gen_img.shape)  # 将一维索引转换为多维索引
            mutated_individual[pixel_idx] = np.random.randint(0, 256)  # 在0到255之间生成随机像素值

    return mutated_individual

# Register the custom mutation function to the toolbox
toolbox.register("mutate", mutate_image, mutation_rate=0.05)
toolbox.register("select", tools.selNSGA2)

# Load image paths
img_paths = image.list_pictures('./testing/center', ext='jpg')

# Create Keras functions for each objective
iterate1 = K.function([input_tensor], [loss1])
iterate2 = K.function([input_tensor], [loss1_neuron])

# Create a population
population = toolbox.population(n=POPULATION_SIZE)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Evolve the population
for gen in range(GENERATIONS):
    offspring = toolbox.select(population, len(population))

    offspring1 = []
    for child1, child2, idx in zip(offspring[::2], offspring[1::2], range(len(offspring) // 2)):
        # Load the images corresponding to the individuals
        img_path1 = child1
        img_path2 = child2
        gen_img1 = preprocess_image(img_path1)
        gen_img2 = preprocess_image(img_path2)

        # Mate the individuals
        if random.random() < 0.5:
            # Perform crossover operation on image data
            crossover_point = random.randint(0, gen_img1.shape[0])
            temp_img1 = gen_img1.copy()
            gen_img1[crossover_point:] = gen_img2[crossover_point:]
            gen_img2[crossover_point:] = temp_img1[crossover_point:]

            # Perform mutation operation on image data
            if random.random() < 0.2:
                mutation_row = random.randint(0, gen_img1.shape[1] - 1)
                mutation_col = random.randint(0, gen_img1.shape[2] - 1)
                mutation_channel = random.randint(0, gen_img1.shape[3] - 1)
                gen_img1[0, mutation_row, mutation_col, mutation_channel] = np.random.uniform(0, 255)
            if random.random() < 0.2:
                mutation_row = random.randint(0, gen_img2.shape[1] - 1)
                mutation_col = random.randint(0, gen_img2.shape[2] - 1)
                mutation_channel = random.randint(0, gen_img2.shape[3] - 1)
                gen_img2[0, mutation_row, mutation_col, mutation_channel] = np.random.uniform(0, 255)

            # Update individuals with mutated image data
            child1, child2 = img_path1, img_path2  # Update image paths
            del child1.fitness.values
            del child2.fitness.values
            offspring1.append(child1)
            offspring1.append(child2)
            # Save new individuals
            imsave(f"evolve/evolve{idx}.jpg", deprocess_image(gen_img1[0]).astype('uint8'))
            imsave(f"evolve/evolve{idx + 1}.jpg", deprocess_image(gen_img2[0]).astype('uint8'))
    #print("offspring1:", len(offspring1))

    offspring2 = []
    for mutant, idx in zip(offspring, range(len(offspring))):
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            offspring2.append(mutant)  # 将变异后的个体添加到offspring2中
            #print("mutant:", mutant)
            # Save new individual
            imsave(f"evolve1/evolve{idx}.jpg", deprocess_image(preprocess_image(mutant)[0]).astype('uint8'))
            #imsave(f"evolve1/evolve{idx}.jpg", preprocess_image(mutant)[0])
    #print("offspring2:", len(offspring2))

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring
    #population += offspring1
    #population += offspring2  # 将变异后的后代种群添加到种群中

print("len(population):",len(population))
# Get the Pareto front
pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=False)
pareto_front = [ind for front in pareto_fronts for ind in front]
top_pareto = pareto_front[:10]
'''pareto_front = tools.sortNondominated(population, len(population), first_front_only=False)[0]
print("pareto_front:", pareto_front)'''

# Print the solutions in the Pareto front
for ind in pareto_front:
    print("Image Path:", ind)
    print("Loss1:", -ind.fitness.values[0])
    print("Loss1 Neuron:", -ind.fitness.values[1])
print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f'
      % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)
print(len(top_pareto))

# 提取帕累托前沿数据
loss1_values = [-ind.fitness.values[0] for ind in top_pareto]
loss1_neuron_values = [-ind.fitness.values[1] for ind in top_pareto]

# 绘制帕累托前沿图
plt.figure(figsize=(8, 6))
plt.scatter(loss1_values, loss1_neuron_values, color='b')
plt.xlabel('Loss1')
plt.ylabel('Loss1 Neuron')
plt.title('Pareto Front')
plt.grid(True)
plt.show()