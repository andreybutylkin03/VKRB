import numpy as np
import data_simple
from xml.etree import ElementTree as et


def stand_parser(pc_num, cluster_num, data):
    tree = et.parse(f"data/stand_{pc_num}_{cluster_num}.xml")

    stand = tree.getroot()

    PC = stand.find("PC")
    Cluster = stand.find("Cluster")

    V = dict()
    mbps_to_base = 1000*1000/8

    for pc in PC:
        V[((0, int(pc.get("id"))), (1, 0))] = np.float64(pc.get("V")) * mbps_to_base

    for cluster in Cluster:
        V[((1, 0), (2, int(cluster.get("id"))))] = np.float64(cluster.get("V_to")) * mbps_to_base
        V[((2, int(cluster.get("id"))), (1, 0))] = np.float64(cluster.get("V_out")) * mbps_to_base

    file_on_link = dict()

    for i in range(pc_num):
        file_on_link[((0, i), (1, 0))] = set()

    for i in range(cluster_num):
        file_on_link[((2, i), (1, 0))] = set()
        file_on_link[((1, 0), (2, i))] = set()

    for i, task in data.task.items():
        for file_id_dep in task.file_dep:
            if data.file[file_id_dep].source != (2, task.cluster):
                if data.file[file_id_dep].source != (1, 0):
                    file_on_link[(data.file[file_id_dep].source, (1,0))].add(file_id_dep)
             
                file_on_link[((1, 0), (2, task.cluster))].add(file_id_dep)

    event_num = len(data.task)

    for val in file_on_link.values():
        event_num += len(val)

    offset = np.int64(0)

    link = dict()

    for key, val in file_on_link.items():
        if len(val) == 0:
            continue

        link[key] = data_simple.Link(key, V[key], val, offset)
        offset += len(val) * event_num

    return (link, event_num)
