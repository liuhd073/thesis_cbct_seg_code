import re
import os

s = m42.scan[6]
patient = s.url[int(re.search(r":[0-9]+\.", s.url).start())+1:int(re.search(r":[0-9]+\.", s.url).end())-1]
print(patient)

bladder = s.alias.lower() + "_bladder" + "_pdr" # Use for CBCT segmentation
#bladder = "99refscan_bladder_pdr_" # Use for CT segmentation
print(bladder)
bladder = m42.StructureSet[1].GetByName(bladder)

cervix = s.alias.lower() + "_cervix/uterus" + "_pdr" # Use for CBCT segmentation
#cervix = "99refscan_cervix/uterus_pdr" # Use for CT segmentation
print(cervix)
cervix = m42.StructureSet[1].GetByName(cervix)
print(cervix.volume())
seg_bladder = s.burnto(bladder, 1, True, True)
seg_cervix = s.burnto(cervix, 1, True, True)
AddScan() 
m42.scan[7] = seg_bladder
AddScan()
m42.scan[8] = seg_cervix

print(r'D:\data\cervix\\' + patient + '\\bladder_' + s.alias.lower())
m42.scan[7].writenifti(r'D:\data\cervix\patients\\' + patient + '\\bladder_' + s.alias.lower())
m42.scan[8].writenifti(r'D:\data\cervix\patients\\' + patient + '\\cervix_uterus_' + s.alias.lower())

