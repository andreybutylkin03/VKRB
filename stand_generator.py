import numpy as np
from xml.etree import ElementTree as et


if __name__ == "__main__":
    pc_num = int(input())
    cluster_num = int(input())

    stand = et.Element("Stand")
    tree = et.ElementTree(stand)
    
    PC = et.SubElement(stand, "PC", {"n":str(pc_num)})
    Cluster = et.SubElement(stand, "Cluster", {"n":str(cluster_num)})

    for i in range(pc_num):
        vr_pc = et.SubElement(PC, "pc", {"id":str(i), "V": str(abs(np.random.normal(300, 200)))})

    for i in range(cluster_num):
        vr_cluster = et.SubElement(Cluster, "cluster", {"id":str(i), "V_to": str(abs(np.random.normal(500, 100))), \
                "V_out": str(abs(np.random.normal(500, 100)))})

    tree.write(f"data/stand_{pc_num}_{cluster_num}.xml", encoding="utf-8", xml_declaration=True)
