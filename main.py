from cerebellum import Cerebellum
from copy import deepcopy as dc

#input size will be the number of sensors I got
Cere = Cerebellum()
Cere2= dc(Cere)

print(Cere.DCN.activity)
print(Cere2.DCN.activity)
Cere.test_change_thing()
print(Cere.DCN.activity)
print(Cere2.DCN.activity)