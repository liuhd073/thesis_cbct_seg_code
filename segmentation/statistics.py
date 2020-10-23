import pickle
import numpy as np

statistics = pickle.load(open("image_statistics.p", 'rb'))

files = list(statistics.keys())[2:]

means = []
norm_means = []
mins = []
maxs = []
for f in files:
    means.append(statistics[f]["mean"])
    norm_means.append(statistics[f]["norm_mean"])
    mins.append(statistics[f]["min"])
    maxs.append(statistics[f]["max"])

print()
print("CBCT dataset CTs:")
print("mean:", np.mean(means))
print("normalized mean:", np.mean(norm_means))
print("min:", np.min(mins))
print("max:", np.max(maxs))



statistics = pickle.load(open("image_statistics_extra.p", 'rb'))

files = list(statistics.keys())[2:]

means = []
norm_means = []
mins = []
maxs = []
for f in files:
    means.append(statistics[f]["mean"])
    norm_means.append(statistics[f]["norm_mean"])
    mins.append(statistics[f]["min"])
    maxs.append(statistics[f]["max"])

print()
print("extra dataset CTs:")
print("mean:", np.mean(means))
print("normalized mean:", np.mean(norm_means))
print("min:", np.min(mins))
print("max:", np.max(maxs))


