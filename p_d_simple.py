from data_simple import *
import numpy as np

def rand_0_2048(num_point: np.int64) -> NDArray[np.float64]:
    if num_point == 1:
        return np.array([2048], dtype=np.float64)
    pre_p = np.random.randint(0, 2048, num_point-1)
    pre_p.sort()
    p = np.empty(shape=(num_point), dtype=np.float64)
    p[0] = pre_p[0]
    for i in np.arange(1, num_point-1):
        p[i] = pre_p[i] - pre_p[i-1]
    p[num_point-1] = 2048 - pre_p[num_point-2]

    return p

def p_d_produce(link: dict[tuple[tuple[int, int], tuple[int, int]], Link], \
        event_num: np.int64, num_prod: np.int32, offset_size: np.int64) -> NDArray[np.float64]:
    p = np.empty(shape=(num_prod, offset_size), dtype = np.float64)

    link_num = len(link)
    offset = np.empty(shape=(len(link)), dtype=np.int64)
    num_file = np.empty(shape=(len(link)), dtype=np.int64)
    for i, value in enumerate(link.values()):
        offset[i] = value.offset
        num_file[i] = value.num_file

    for i in range(num_prod):
        for j in range(link_num):
            for k in range(event_num):
                if (num_file[j]):
                    p[i, offset[j]+k*num_file[j]:offset[j]+(k+1)*num_file[j]] = rand_0_2048(num_file[j])

    return p

def A_produce(link: dict[tuple[tuple[int, int], tuple[int, int]], Link], \
        event_num: np.int64, offset_size: np.int64):
    A = np.zeros(shape=(len(link)*event_num, offset_size), dtype=np.float64)    
    b = np.ones(len(link)*event_num, dtype=np.float64)
    bounds = [(np.float64(0), np.float64(1)) for i in range(offset_size)]

    link_num = len(link)
    offset = np.empty(shape=(len(link)), dtype=np.int64)
    num_file = np.empty(shape=(len(link)), dtype=np.int64)

    for i, value in enumerate(link.values()):
        offset[i] = value.offset
        num_file[i] = value.num_file

    i = 0

    for j in range(link_num):
        for k in range(event_num):
            if (num_file[j]):
                A[i, offset[j]+k*num_file[j]:offset[j]+(k+1)*num_file[j]] = np.ones(num_file[j])
                i += 1

    return [A, b, bounds]
