import json
import os
from matplotlib import pyplot as plt


DATA_DIR = "/mnt/ssd2/AllDatasets/ProstateDataset/Level1_128_rich"
TRAIN = os.path.join(DATA_DIR,"train")
MASK = os.path.join(DATA_DIR,"mask")
LABEL = os.path.join(DATA_DIR,"Label")
LABEL_LIST =  [ os.path.join(LABEL,fname) for fname in os.listdir(LABEL) ]

OUT_DIR = "/mnt/ssd2/Projects/ProstateChallenge/output/DatasetInfo"

stats_BC={"Benign_comb":0,"Cancerous_comb":0,"Cancerous_Kar":0,"G3":0,"G4":0,"G5":0}

for fname in LABEL_LIST:
    with open(fname,"r") as f:
        dict = json.loads(f.read())
    # iterate through patch level
    if dict["data_provider"] == "karolinska":
        for i in range(dict["patches_num"]):
            if dict["patches_stat"]["cancerous_tissue_perc"][i]> 0.001:
                stats_BC["Cancerous_Kar"]+=1
            else:
                stats_BC["Benign_comb"] += 1
    else:
        for i in range(dict["patches_num"]):
            if dict["patches_stat"]["Gleason_3_perc"][i]> 0.001:
                stats_BC["G3"]+=1
            elif dict["patches_stat"]["Gleason_4_perc"][i]> 0.001:
                stats_BC["G4"]+=1
            elif dict["patches_stat"]["Gleason_5_perc"][i]> 0.001:
                stats_BC["G5"]+=1
            else:
                stats_BC["Benign_comb"]+=1

stats_BC["Cancerous_comb"] = stats_BC["Cancerous_Kar"]+stats_BC["G3"]+stats_BC["G4"]+stats_BC["G5"]
plt.bar(stats_BC.keys(), stats_BC.values())
plt.title("patches stats for {}".format(DATA_DIR.split("/")[-1]))
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='center')

plt.savefig(OUT_DIR + "/" +"patches stats for {}".format(DATA_DIR.split("/")[-1]) )
plt.show()