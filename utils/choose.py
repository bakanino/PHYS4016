from math import comb

A = (comb(60,7)*comb(30,3))/(comb(90,10))
B = (comb(30,7)*comb(60,3))/(comb(90,10))

print(A/(A+B))

