import copy
import numpy as np
from numpy.typing import NDArray
from heapq import *
from data_simple import *


class T:
    def __init__(self, data: Data, event_num: np.int64, 
                 zero_link: dict[tuple[tuple[int, int], tuple[int, int]], Link]):
        self.data = data
        self.event_num = event_num
        self.zero_link = zero_link # Dict[tuple[tuple[int, int], tuple[int, int]], Link] 

    def V_calc(self, p: NDArray[np.float64]) -> dict[tuple[tuple[int, int], tuple[int, int]], NDArray[np.float64]]:
        V_f_file = {key: np.empty(self.event_num*link.link.num_file, dtype=np.float64) \
                    for key, link in self.link.items()}

        for key, link in self.link.items():
            for file_id_cnt in link.file_work.keys():
                for event_count in range(self.event_num):
                    V_f_file[key][event_count * link.link.num_file + link.link.file_offset[file_id_cnt]] = \
                        link.link.V * p[link.link.offset + \
                        event_count * link.link.num_file + \
                        link.link.file_offset[file_id_cnt]]

        return V_f_file

    def best_T(self) -> np.float64:
        ans_T = np.float64(np.inf)

        for [[type_in, id_in], [type_out, id_out]], link in self.link.items():
            if type_in == 0 or type_in == 2: # PC to Server or Cluster to Server
                for file_id_cnt in link.file_worklist:
                    file_work_cnt = link.file_work[file_id_cnt]

                    if type_in == 2 and self.task_work[self.data.file[file_id_cnt].task_prod].T == 0:
                        continue

                    if type_in == 2 and self.task_work[self.data.file[file_id_cnt].task_prod].T > self.T:
                        continue

                    T_event = (self.data.file[file_id_cnt].size - file_work_cnt.cur_size) / \
                            self.V_f_file[((type_in, id_in), (type_out, id_out))]\
                            [self.event_count * link.link.num_file + link.link.file_offset[file_id_cnt]]

                    if T_event < ans_T:
                        ans_T = T_event

            elif type_in == 1: # Server to Cluster
                for file_id_cnt in link.file_worklist:
                    file_work_cnt = link.file_work[file_id_cnt]

                    if self.data.file[file_id_cnt].source[0] == 1: # if file source is file server
                        T_event = (self.data.file[file_id_cnt].size - file_work_cnt.cur_size) / \
                                self.V_f_file[((type_in, id_in), (type_out, id_out))]\
                                [self.event_count * link.link.num_file + link.link.file_offset[file_id_cnt]]

                        if T_event < ans_T:
                            ans_T = T_event
                    else: # if file source is pc or another cluster
                        task_prod = self.data.file[file_id_cnt].task_prod

                        if task_prod != -1 and self.task_work[task_prod].T == 0: 
                            continue

                        if task_prod != -1 and self.task_work[task_prod].T > self.T:
                            continue

                        # if file don't download on server skip
                        prev_link = self.link[(self.data.file[file_id_cnt].source, (1, 0))]
                        if file_id_cnt in prev_link.file_worklist:
                            continue

                        T_event = (self.data.file[file_id_cnt].size - file_work_cnt.cur_size) / \
                                self.V_f_file[((type_in, id_in), (type_out, id_out))]\
                                [self.event_count * link.link.num_file + link.link.file_offset[file_id_cnt]]

                        if T_event < ans_T:
                            ans_T = T_event

        return self.T + ans_T

    def remove_cur_file_dep(self, file_id):
        for task in self.data.file[file_id].tasks_need:
            self.task_work[task].cur_file_dep.discard(file_id)

    def time_up(self, new_T: np.float64):
        count_event = 0 # num event in time up

        for [[type_in, id_in], [type_out, id_out]], link in self.link.items():
            del_file_id = set()

            if type_in == 0 or type_in == 2: # PC to Server or Cluster to Server
                for file_id_cnt in link.file_worklist:
                    file_work_cnt = link.file_work[file_id_cnt]

                    if type_in == 2 and self.task_work[self.data.file[file_id_cnt].task_prod].T == 0:
                        continue

                    if type_in == 2 and self.task_work[self.data.file[file_id_cnt].task_prod].T > self.T:
                        continue

                    if len(file_work_cnt.file_part) and file_work_cnt.file_part[-1][1] == new_T:
                        continue

                    file_part = self.V_f_file[((type_in, id_in), (type_out, id_out))]\
                        [self.event_count * link.link.num_file + link.link.file_offset[file_id_cnt]] * (new_T - self.T)
                    
                    file_part = np.floor(file_part).astype(np.int64)

                    if abs(file_work_cnt.cur_size + file_part - self.data.file[file_id_cnt].size) < 10:
                        file_part = self.data.file[file_id_cnt].size - file_work_cnt.cur_size
                        file_work_cnt.cur_size = self.data.file[file_id_cnt].size
                        file_work_cnt.file_part.append([file_part, new_T])

                        file_work_cnt.t_end = new_T
                        count_event += 1
                        #print("event: ", self.event_count, [[type_in, id_in], [type_out, id_out]], '-', file_id_cnt)
                        #print("from ", self.T, " to ", new_T)
                        del_file_id.add(file_id_cnt)
                    else:
                        file_work_cnt.cur_size += file_part 
                        file_work_cnt.file_part.append([file_part, new_T])

            elif type_in == 1: # Server to Cluster
                for file_id_cnt in link.file_worklist:
                    file_work_cnt = link.file_work[file_id_cnt]

                    if self.data.file[file_id_cnt].source[0] == 1: # if file source is file server
                        file_part = self.V_f_file[((type_in, id_in), (type_out, id_out))]\
                            [self.event_count * link.link.num_file + link.link.file_offset[file_id_cnt]] * \
                            (new_T - self.T)
                        
                        file_part = np.floor(file_part).astype(np.int64)

                        if abs(file_work_cnt.cur_size + file_part - self.data.file[file_id_cnt].size) < 10:
                            file_part = self.data.file[file_id_cnt].size - file_work_cnt.cur_size
                            file_work_cnt.cur_size = self.data.file[file_id_cnt].size
                            file_work_cnt.file_part.append([file_part, new_T])

                            file_work_cnt.t_end = new_T
                            count_event += 1

                            #print("event: ", self.event_count, [[type_in, id_in], [type_out, id_out]], '-', file_id_cnt)
                            #print("from ", self.T, " to ", new_T)
                            self.remove_cur_file_dep(file_id_cnt)
                            del_file_id.add(file_id_cnt)
                        else:
                            file_work_cnt.cur_size += file_part 
                            file_work_cnt.file_part.append([file_part, new_T])

                    else: # if file source is pc or another cluster
                        task_prod = self.data.file[file_id_cnt].task_prod

                        if task_prod != -1 and self.task_work[task_prod].T == 0: 
                            continue

                        if task_prod != -1 and self.task_work[task_prod].T > self.T:
                            continue

                        # first what we need to do it is upping time for file in source link
                        prev_link = self.link[(self.data.file[file_id_cnt].source, (1, 0))]

                        prev_file_work_cnt = prev_link.file_work[file_id_cnt]

                        if file_id_cnt in prev_link.file_worklist and prev_file_work_cnt.file_part[-1][1] != new_T:
                            prev_file_part = self.V_f_file[(self.data.file[file_id_cnt].source, (1, 0))]\
                                [self.event_count * prev_link.link.num_file + prev_link.link.file_offset[file_id_cnt]] * \
                                (new_T - self.T)

                            prev_file_part = np.floor(prev_file_part).astype(np.int64)

                            if abs(prev_file_work_cnt.cur_size + prev_file_part - self.data.file[file_id_cnt].size) < 10:
                                prev_file_part = self.data.file[file_id_cnt].size - prev_file_work_cnt.cur_size
                                prev_file_work_cnt.cur_size = self.data.file[file_id_cnt].size
                                prev_file_work_cnt.file_part.append([prev_file_part, new_T])

                                prev_file_work_cnt.t_end = new_T
                                count_event += 1

                                #print("event: ", self.event_count, (self.data.file[file_id_cnt].source, (1, 0)), '-', file_id_cnt)
                                #print("from ", self.T, " to ", new_T)
                                prev_link.file_worklist.discard(file_id_cnt)
                            else:
                                prev_file_work_cnt.cur_size += prev_file_part 
                                prev_file_work_cnt.file_part.append([prev_file_part, new_T])

                        file_part = self.V_f_file[((type_in, id_in), (type_out, id_out))]\
                            [self.event_count * link.link.num_file + link.link.file_offset[file_id_cnt]] * \
                            (new_T - self.T)
                        
                        file_part = np.floor(file_part).astype(np.int64)

                        if file_work_cnt.cur_size + file_part > prev_file_work_cnt.cur_size:
                            file_part = prev_file_work_cnt.cur_size - file_work_cnt.cur_size

                        if abs(file_work_cnt.cur_size + file_part - self.data.file[file_id_cnt].size) < 10:
                            file_part = self.data.file[file_id_cnt].size - file_work_cnt.cur_size
                            file_work_cnt.cur_size = self.data.file[file_id_cnt].size
                            file_work_cnt.file_part.append([file_part, new_T])

                            file_work_cnt.t_end = new_T
                            count_event += 1
                            #print("event: ", self.event_count, [[type_in, id_in], [type_out, id_out]], '-', file_id_cnt)
                            #print("from ", self.T, " to ", new_T)

                            self.remove_cur_file_dep(file_id_cnt)
                            del_file_id.add(file_id_cnt)
                        else:
                            file_work_cnt.cur_size += file_part 
                            file_work_cnt.file_part.append([file_part, new_T])

            link.file_worklist -= del_file_id

        self.event_count += count_event

    def t_simple(self, pp: NDArray[np.float64]) -> np.float64:
        p = copy.deepcopy(pp)

        for i in np.arange(p.shape[0]):
            if p[i] == 0:
                p[i] += np.finfo(np.float64).eps

        self.link = {key: Link_Work(link, self.data) for key, link in self.zero_link.items()} 
        self.V_f_file = self.V_calc(p)
        #print(self.V_f_file)
        self.task_worklist = set(self.data.task.keys())
        self.task_work = {key: Task_Work(value) for key, value in self.data.task.items()} # Dict[int, Task_Work]
        self.T = np.float64(0)
        self.event_count = 0

        # delete source file == source task
        for task in self.data.task.keys():
            file_del = set()

            for file in self.task_work[task].cur_file_dep:
                if self.data.file[file].source == (2, self.data.task[task].cluster):
                    file_del.add(file)

            self.task_work[task].cur_file_dep -= file_del

        exec_task = [] # set of (T) -- task in exec
        ij = 0

        while self.event_count != self.event_num:
            del_task_id = set()
            ij += 1

            if ij > self.event_num:
                break

            for task in self.task_worklist:
                if not len(self.task_work[task].cur_file_dep):
                    del_task_id.add(task)
                    self.task_work[task].T = self.T + self.task_work[task].task.exec_time
                    heappush(exec_task, self.task_work[task].T)

            self.task_worklist -= del_task_id 

            best_T = self.best_T() # new T
            #print('best_T: ', best_T)
            #print('exec_task: ', exec_task)
            #print("cur_file_dep ", self.task_work[0].cur_file_dep)

            if (len(exec_task) == 0 or exec_task[0] > best_T):
                #print("REQ")
                self.time_up(best_T)
                self.T = best_T
            else:
                #print("TASK")
                self.time_up(exec_task[0])
                self.T = exec_task[0]
                while len(exec_task) and exec_task[0] == self.T:
                    heappop(exec_task)
                    self.event_count += 1
            #print(self.event_count)

        return self.T 
