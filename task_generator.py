import numpy as np
from xml.etree import ElementTree as et


if __name__ == "__main__":
    pc_num = int(input())
    cluster_num = int(input())
    task_num = int(input())
    file_num = int(input())

    pcs_f_num = file_num // 3
    c_f_num = file_num - pcs_f_num
    pc_f_num = pcs_f_num // 3
    s_f_num = pcs_f_num - pc_f_num
    mean_task_prod = (c_f_num + task_num - 1) // task_num + 0.3
    std_task_prod = 1 

    d_task_file_produce = dict() # Dict[id_task, [id_start, id_end]] 
    d_task_file_dep = dict() # Dict[id_task, [{id_task}, {id_file}]]
    d_file_task_prod = dict() # Dict[id_file, id_task]
    d_task_cluster = dict() # Dict[id_task, id_cluster]

    for i in range(task_num):
        d_task_cluster[i] = np.random.randint(0, cluster_num)

    for i in range(pcs_f_num):
        d_file_task_prod[i] = -1

    id_cur = pcs_f_num
    for i in range(task_num):
        if id_cur == file_num:
            d_task_file_produce[i] = [id_cur, id_cur]
            continue

        count_file = int(np.round(abs(np.random.normal(mean_task_prod, std_task_prod))))
        id_end = id_cur + count_file 
        id_end = min(id_end, file_num)
        if i == task_num-1:
            id_end = file_num
        d_task_file_produce[i] = [id_cur, id_end]
        for j in range(id_cur, id_end):
            d_file_task_prod[j] = i
        id_cur = id_end

    for i in range(task_num):
        count_file = max(int(np.round(abs(np.random.normal(file_num/task_num, 1)))), 1)
        id_file = set(np.random.randint(0, d_task_file_produce[i][0], size=count_file))
        id_task = set()
        for j in id_file:
            id_task |= {d_file_task_prod[j]}
        id_task -= {-1}
        d_task_file_dep[i] = [id_task, id_file]

    task_flow = et.Element("Task_Flow")
    tree = et.ElementTree(task_flow)
    
    File = et.SubElement(task_flow, "File", {"n":str(file_num)})
    Task = et.SubElement(task_flow, "Task", {"n":str(task_num)})

    for i in range(pc_f_num):
        vr_id_pc = np.random.randint(0, pc_num)
        vr_file = et.SubElement(File, "file", {"id":str(i),\
                "size": str(int(np.random.uniform(10**9, 4*10**10))),\
                "source": f"0,{vr_id_pc}",\
                "task_prod": "-1"})

    for i in range(pc_f_num, pcs_f_num):
        vr_file = et.SubElement(File, "file", {"id":str(i),\
                "size": str(int(np.random.uniform(10**9, 4*10**10))),\
                "source": f"1,0",\
                "task_prod": "-1"})

    for i in range(pcs_f_num, file_num):
        vr_file = et.SubElement(File, "file", {"id":str(i),\
                "size": str(int(np.random.uniform(10**9, 4*10**10))),\
                "source": f"2,{d_task_cluster[d_file_task_prod[i]]}",\
                "task_prod": f"{d_file_task_prod[i]}"})

    for i in range(task_num):
        vr_prod = set([i for i in range(d_task_file_produce[i][0], d_task_file_produce[i][1])])
        vr_task = et.SubElement(Task, "task", {"id":str(i), "cluster":f"{d_task_cluster[i]}", \
                "task_dep":str(d_task_file_dep[i][0]), "file_dep":str(d_task_file_dep[i][1]), \
                "file_prod":str(vr_prod), "exec_time":str(np.random.randint(60, 10*60))})

    tree.write(f"data/task_flow_{pc_num}_{cluster_num}_{task_num}_{file_num}.xml", encoding="utf-8", xml_declaration=True)
