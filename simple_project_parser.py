import numpy as np
import data_simple
from xml.etree import ElementTree as et


def project_parser(pc_num, cluster_num, task_num, file_num):
    tree = et.parse(f"data/task_flow_{pc_num}_{cluster_num}_{task_num}_{file_num}.xml")

    task_flow = tree.getroot()

    File = task_flow.find("File")
    Task = task_flow.find("Task")

    task = dict()
    file = dict()
    tasks_need = dict()

    for vr_task in Task:
        task[int(vr_task.get("id"))] = data_simple.Task(int(vr_task.get("id")), int(vr_task.get("cluster")), eval(vr_task.get("task_dep")), \
                eval(vr_task.get("file_dep")), eval(vr_task.get("file_prod")), int(vr_task.get("exec_time")))
        for id_file in task[int(vr_task.get("id"))].file_dep:
            tasks_need.setdefault(id_file, set())
            tasks_need[id_file].add(int(vr_task.get("id")))

    print(tasks_need)
    for vr_file in File:
        if int(vr_file.get("id")) not in tasks_need.keys():
            file[int(vr_file.get("id"))] = data_simple.File(int(vr_file.get("id")), int(vr_file.get("size")), \
                tuple(map(int, vr_file.get("source").split(','))), int(vr_file.get("task_prod")), set())
        else:
            file[int(vr_file.get("id"))] = data_simple.File(int(vr_file.get("id")), int(vr_file.get("size")), \
                tuple(map(int, vr_file.get("source").split(','))), int(vr_file.get("task_prod")),\
                tasks_need[int(vr_file.get("id"))])

    data = data_simple.Data(task, file)

    return data
