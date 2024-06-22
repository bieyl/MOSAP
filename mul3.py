from deap import base, creator, tools
from driving_models import *
from utils import *
from keras.preprocessing.image import *
from imageio import imsave
import matplotlib.pyplot as plt
import torch
from differential_color_functions import *

# Set device
device = torch.device("cuda:0")

# Define problem-specific parameters
POPULATION_SIZE = 100
GENERATIONS = 5
ERROR_THRESHOLD = 0.01  # Prediction error threshold

# Image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols,3)

# Define the input tensor
input_tensor = Input(shape=input_shape)

# Load the model and initialize coverage tables
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model_layer_dict1 = init_coverage_tables1(model1)
layer_name1, index1 = neuron_to_cover(model_layer_dict1)

# Define functions to calculate loss1, loss1_neuron, and color difference
loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])

def color_diff(inputs, adv_input, device):
    #print("inputs:",inputs)
    #print("adv_input:",adv_input)
    inputs = np.transpose(inputs, (0, 3, 1, 2))  # 将通道维度移到第二个位置
    batch_size = inputs.shape[0]
    #print("inputs.shape:", inputs.shape)
    inputs = np.clip(inputs, 0, 1)
    inputs_LAB = rgb2lab_diff(inputs, device)

    adv_input = np.transpose(adv_input, (0, 3, 1, 2))  # 将通道维度移到第二个位置

    adv_input = np.clip(adv_input, 0, 1)

    adv_input_LAB = rgb2lab_diff(adv_input, device)
    print("inputs range:", inputs.min(), inputs.max())
    print("adv_input range:", adv_input.min(), adv_input.max())
    d_map = ciede2000_diff(inputs_LAB, adv_input_LAB, device).unsqueeze(1)
    l2 = torch.norm(d_map.view(batch_size, -1), dim=1)
    print("l2:",l2)
    return l2

# Define the problem-specific function for evaluating individuals
def evaluate(individual, adv_individual):
    # Load the original and adversarial images
    img_path = individual
    gen_img = preprocess_image(img_path)
    #print("gen_img:",gen_img)

    adv_path = adv_individual
    adv_img = preprocess_image(adv_path)
    #print("adv_img:",adv_img)

    # Calculate the objectives for the adversarial image
    loss1_value = -iterate1([adv_img])[0]
    loss1_neuron_value = -iterate2([adv_img])[0]

    # Calculate color difference between original and adversarial images
    loss_color = color_diff(gen_img, adv_img, device)

    # Calculate the prediction error
    pred_error = loss1_value + ERROR_THRESHOLD

    # Individuals whose prediction error exceeds the threshold are assigned a very large fitness value
    if pred_error > 0:
        fitness = (100.0, 100.0, 100.0)  # Large fitness value indicates unsatisfied constraints
    else:
        fitness = (loss1_value, loss1_neuron_value, loss_color)
        update_coverage(adv_img, model1, model_layer_dict1)

    # Return a tuple of objectives
    return fitness

# Create a fitness class with multiple objectives
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

# Create an individual class
creator.create("Individual", str, fitness=creator.FitnessMulti)

# Initialize the toolbox
toolbox = base.Toolbox()

# Define functions for creating individuals and populations
def create_individual():
    path = random.choice(all_paths)
    ind = creator.Individual(path)
    return ind

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the necessary functions
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)

# Define the mutation function
def mutate_image(individual, mutation_rate=0.05):
    gen_img = preprocess_image(individual)

    mutated_individual = np.copy(gen_img)

    if np.random.rand() < mutation_rate:
        num_pixels_to_mutate = int(mutation_rate * gen_img.size)
        mutation_indices = np.random.choice(range(gen_img.size), num_pixels_to_mutate, replace=False)

        for idx in mutation_indices:
            pixel_idx = np.unravel_index(idx, gen_img.shape)
            mutated_individual[pixel_idx] = np.random.randint(0, 256)

    return mutated_individual

toolbox.register("mutate", mutate_image, mutation_rate=0.05)
toolbox.register("select", tools.selNSGA2)

# Load image paths
img_paths = image.list_pictures('./testing/center', ext='jpg')
all_paths = img_paths.copy()

# Create Keras functions for each objective
iterate1 = K.function([input_tensor], [loss1])
iterate2 = K.function([input_tensor], [loss1_neuron])

# Create a population
population = toolbox.population(n=POPULATION_SIZE)

# Evolve the population
for gen in range(GENERATIONS):
    print("11111111111111111111111111111111111111111111111111111111")
    print("population:", population)
    offspring = toolbox.select(population, len(population))

    offspring_paths = []
    idx_counter = 0

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # Load the images corresponding to the individuals
        img_path1 = child1
        img_path2 = child2
        gen_img1 = preprocess_image(img_path1)
        gen_img2 = preprocess_image(img_path2)

        # Perform crossover operation
        if random.random() < 0.5:
            crossover_point = random.randint(0, gen_img1.shape[0])
            temp_img1 = gen_img1.copy()
            gen_img1[crossover_point:] = gen_img2[crossover_point:]
            gen_img2[crossover_point:] = temp_img1[crossover_point:]

        # Perform mutation operation
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

        # Save new individuals
        new_path1 = f"./mutate_image/{idx_counter}.jpg"
        imsave(new_path1, deprocess_image(gen_img1[0]).astype('uint8'))
        offspring_paths.append((img_path1, new_path1))
        all_paths.append(new_path1)
        del img_path1.fitness.values  # 删除适应度值
        idx_counter += 1

        new_path2 = f"./mutate_image/{idx_counter}.jpg"
        imsave(new_path2, deprocess_image(gen_img2[0]).astype('uint8'))
        offspring_paths.append((img_path2, new_path2))
        all_paths.append(new_path2)
        del img_path2.fitness.values  # 删除适应度值
        idx_counter += 1

    # Check for additional mutations
    for mutant in offspring[len(offspring)//2:]:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            mutant_path = f"./mutate_image/{idx_counter}.jpg"
            imsave(mutant_path, deprocess_image(preprocess_image(mutant)[0]).astype('uint8'))
            offspring_paths.append((mutant, mutant_path))
            all_paths.append(mutant_path)
            idx_counter += 1

    #print("offspring_paths:", offspring_paths)
    # 主循环中更新种群部分
    population[:] = [creator.Individual(new_path) for _, new_path in offspring_paths]
    # 计算整个种群的适应度值
    fitnesses = list(map(lambda x: toolbox.evaluate(x[0], x[1]), offspring_paths))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print("2222222222222222222222222222222222222222222222222222222222222")

print("len(population):", len(population))

# Get the Pareto front
pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=False)
pareto_front = [ind for front in pareto_fronts for ind in front]
top_pareto = pareto_front[:20]

# Print the solutions in the Pareto front
for ind in pareto_front:
    print("Image Path:", ind)  # 提取图像路径
    print("Loss1:", -ind.fitness.values[0])  # 提取 Loss1
    print("Loss1 Neuron:", -ind.fitness.values[1])  # 提取 Loss1 Neuron
    #print("Loss Color:", ind.fitness.values[2].item())  # 提取 Color Difference
    # 打印 Loss Color
    loss_color = ind.fitness.values[2]
    if isinstance(loss_color, torch.Tensor):
        print("Loss Color:", loss_color.item())
    else:
        print("Loss Color:", loss_color)
print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f'
      % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)
print(len(top_pareto))

# 提取帕累托前沿数据
loss1_values = [-ind.fitness.values[0] for ind in top_pareto]
loss1_neuron_values = [-ind.fitness.values[1] for ind in top_pareto]
#color_diff_values = [ind.fitness.values[2].item() for ind in top_pareto]  # 修正颜色差值提取方式
color_diff_values = [ind.fitness.values[2].item() if isinstance(ind.fitness.values[2], torch.Tensor) else ind.fitness.values[2] for ind in top_pareto]  # 判断数据类型

# 绘制帕累托前沿图
fig = plt.figure(figsize=(12, 10))  # 增加图像尺寸
ax = fig.add_subplot(111, projection='3d')
ax.scatter(loss1_values, loss1_neuron_values, color_diff_values, c='b', marker='o')
ax.set_xlabel('Loss1')
ax.set_ylabel('Loss1 Neuron')
ax.set_zlabel('Loss Color')
ax.set_title('Pareto Front')

# 提高图像的可读性
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.zaxis.label.set_size(14)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.show()
