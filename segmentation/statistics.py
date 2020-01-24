import pickle
import numpy as np

statistics = pickle.load(open("image_statistics.p", 'rb'))

files = list(statistics.keys())[2:]
print(statistics[files[0]].keys())

means = []
norm_means = []
mins = []
maxs = []
for f in files:
    means.append(statistics[f]["mean"])
    norm_means.append(statistics[f]["norm_mean"])
    mins.append(statistics[f]["min"])
    maxs.append(statistics[f]["max"])

print(np.mean(means), np.mean(norm_means), np.min(mins), np.max(maxs))
