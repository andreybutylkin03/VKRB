import copy
import numpy as np
from numpy.typing import NDArray
from queue import Queue
from heapq import *


class Task:
    def __init__(self, id: int, cluster: int, task_dep: set[int], file_dep: set[int], file_produce: set[int], 
                 exec_time: np.float64):
        self.id = id
        self.cluster = cluster
        self.task_dep = task_dep
        self.file_dep = file_dep
        self.file_produce = file_produce
        self.exec_time = exec_time


class Task_Work:
    def __init__(self, task: Task):
        self.task = task

        self.cur_task_dep = copy.deepcopy(task.task_dep)
        self.cur_file_dep = copy.deepcopy(task.file_dep)
        self.T = np.float64(0)


class File:
    def __init__(self, id: int, size: np.int64, source: tuple[int, int], task_prod: int, tasks_need: set[int]):
        self.id = id
        self.size = size
        self.source = source # 0 - P, 1 - S, 2 - C
        self.task_prod = task_prod # which task id produce this file (-1 if none)
        self.tasks_need = tasks_need # what tasks need this file


class File_Work:
    def __init__(self, file: File, cur_size: np.int64):
        self.file = file

        self.cur_size = cur_size
        self.t_end = np.inf # end of transfer

        self.file_part = list() # Queue[tuple[np.int64, np.float64]] part_size, time
        self.file_part.append([0, 0])


class Data:
    def __init__(self, task: dict[int, Task], file: dict[int, File]):
        self.task = task # Dict[int, Task]
        self.file = file # Dict[int, File]


class Link:
    def __init__(self, id: tuple[tuple[int, int], tuple[int, int]], V: np.float64, file: set[int], offset: np.int64):
        self.id = id
        self.V = V
        self.file = file
        self.offset = offset
        self.num_file = len(self.file)

        self.file_offset = {j:i for i, j in enumerate(self.file)}


class Link_Work:
    def __init__(self, link: Link, data: Data):
        self.link = link

        self.file_work = dict() # Dict[int, File_Work]

        for i in link.file:
            cur_size = np.int64(0)
            
            self.file_work[i] = File_Work(data.file[i], cur_size)

        self.file_worklist = set(self.file_work.keys())
