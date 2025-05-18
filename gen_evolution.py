from T_foo_simple import *
from simple_project_parser import *
from simple_stand_parser import *
from p_d_simple import *
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

POP_SIZE = 500 # Размер популяции
GENERATIONS = 1000  # Число поколений
MUTATION_RATE = 0.2  # Вероятность мутации
CROSSOVER_RATE = 0.9  # Вероятность скрещивания

data = project_parser(5, 5, 20, 80)
link, event_num = stand_parser(5, 5, data)
offset_size = max([link.offset + link.num_file*event_num for link in link.values()])
solve = Solve(data, event_num, link)
link_values = list(link.values())

def objective_function(individual):
    return solve.solve_simple(individual/2048) 

def mutate(individual, mutation_rate=MUTATION_RATE):
    if np.random.rand() < mutation_rate:
        link_m = np.random.randint(0, len(link))

        for i in range(10):
            if link_values[link_m].num_file == 1:
                link_m = np.random.randint(0, len(link))
            else:
                break

        if link_values[link_m].num_file == 1:
            return individual

        for ev in range(event_num):
            new_ind = np.empty(link_values[link_m].num_file)
            new_ind[0] = individual[link_values[link_m].offset + ev*link_values[link_m].num_file]
            for j in range(1, link_values[link_m].num_file):
                new_ind[j] = new_ind[j-1] + individual[link_values[link_m].offset + ev*link_values[link_m].num_file + j]

            file_m = np.random.randint(0, link_values[link_m].num_file-1)
            new_ind[file_m] = np.random.randint(0, 2049)

            cur_i = file_m
            while cur_i and new_ind[cur_i] < new_ind[cur_i-1]:
                new_ind[cur_i], new_ind[cur_i-1] = new_ind[cur_i-1], new_ind[cur_i]
                cur_i -= 1

            while cur_i < link_values[link_m].num_file-1 and new_ind[cur_i] > new_ind[cur_i+1]:
                new_ind[cur_i], new_ind[cur_i+1] = new_ind[cur_i+1], new_ind[cur_i]
                cur_i += 1
            
            prev = 0

            for j in range(link_values[link_m].num_file):
                individual[link_values[link_m].offset + ev*link_values[link_m].num_file + j] = new_ind[j] - prev
                prev = new_ind[j]
            
    return individual

def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        event_m = np.random.randint(0, event_num, len(link))

        for i in range(len(link)):
            beg = link_values[i].offset + event_m[i]*link_values[i].num_file
            end = link_values[i].offset + (event_m[i]+1)*link_values[i].num_file
            parent1[beg:end], parent2[beg:end] = parent2[beg:end], parent1[beg:end]
    return parent1, parent2

def crossover2(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        event_m = np.random.randint(0, event_num)
        link_m = np.random.randint(0, len(link))

        beg = link_values[link_m].offset + event_m*link_values[link_m].num_file

        parent1[beg:], parent2[beg:] = parent2[beg:], parent1[beg:]
        
    return parent1, parent2

def crossover3(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        event_m = np.random.randint(0, event_num, 2)
        link_m = np.random.randint(0, len(link), 2)
        link_m.sort()

        beg = link_values[link_m[0]].offset + event_m[0]*link_values[link_m[0]].num_file
        end = link_values[link_m[1]].offset + event_m[1]*link_values[link_m[1]].num_file
        parent1[beg:end], parent2[beg:end] = parent2[beg:end], parent1[beg:end]
        
    return parent1, parent2

def crossover4(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        event_m = np.random.randint(0, event_num, (len(link), 2))
        event_m.sort(axis=-1)

        for i in range(len(link)):
            beg = link_values[i].offset + event_m[i, 0]*link_values[i].num_file
            end = link_values[i].offset + event_m[i, 1]*link_values[i].num_file
            parent1[beg:end], parent2[beg:end] = parent2[beg:end], parent1[beg:end]
    return parent1, parent2

def crossover5(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point_num = np.random.randint(2, event_num * len(link))
        event_m = np.random.randint(0, event_num, point_num)
        link_m = np.random.randint(0, len(link), point_num)

        fl = True

        for i in range(point_num-1):
            if fl:
                fl = False
                beg = link_values[link_m[i]].offset + event_m[i]*link_values[link_m[i]].num_file
                end = link_values[link_m[i+1]].offset + event_m[i+1]*link_values[link_m[i+1]].num_file
                if beg > end:
                    beg, end = end, beg
                parent1[beg:end], parent2[beg:end] = parent2[beg:end], parent1[beg:end]
            else:
                fl = True

    return parent1, parent2

def select1(population, fitnesses, rate):
    idx = np.argsort(fitnesses)  # Сортируем по убыванию фитнеса
    return population[idx[:int(POP_SIZE * rate)]]  # Берём топ-rate%

def foo_gen(foo_mut, foo_cros, foo_sel, elit_p, popul):
    population = copy.deepcopy(popul)

    res_c = np.empty(GENERATIONS+1, dtype=np.float64)

    for gen in range(GENERATIONS):
        # 1. Оценка приспособленности
        with Pool() as pool:
            fitnesses = np.array(pool.map(objective_function, population))

        res_c[gen] = np.min(fitnesses)
        # 2. Выводим лучшего
        if gen%100==0:
            best_idx = np.argmin(fitnesses)
            print(f"Поколение {gen+1}: Лучший фитнес = {fitnesses[best_idx]:.4f}")
    
        # 3. Отбор элит 
        if elit_p > 0.0:
            elite_size = int(elit_p * POP_SIZE)
            elite_indices = np.argsort(fitnesses)[:elite_size]
            elite = population[elite_indices]
    
        # 4. Отбор родителей
        parents = foo_sel(population, fitnesses, 0.5)
    
        # 5. Cкрещивание
        offspring = []
        for _ in range(len(parents) // 2):
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
            c1, c2 = foo_cros(p1, p2)
            offspring.append(c1)
            offspring.append(c2)
    
        # 6. Мутация
        offspring = np.array([foo_mut(ind) for ind in offspring])
    
        # 7. Создание новой популяции
        if elit_p > 0.0:
            population = np.vstack([elite, offspring[:POP_SIZE - elite_size]])
        else:
            population = offspring
    
    # --- Вывод лучшего решения ---
    fitnesses = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitnesses)
    res_c[GENERATIONS] = fitnesses[best_idx]
    print("Фитнес лучшего:", fitnesses[best_idx])
    return res_c

num_workers = multiprocessing.cpu_count()

population = p_d_produce(link, event_num, POP_SIZE, offset_size)

res_c_1_1_1_0 = foo_gen(mutate, crossover, select1, 0.05, population)
