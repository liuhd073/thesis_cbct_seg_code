import re
import os

print("Start test")

rootdir = r"D:\data\cervix\patients"

for i, s in enumerate(m42.scan):
    # Save scans
    patient = s.url[int(re.search(r":[0-9]+\.", s.url).start())+1:int(re.search(r":[0-9]+\.", s.url).end())-1]
    s.writenifti(rootdir + "\\" + patient + "\\" + s.alias)
    print(s.alias)
