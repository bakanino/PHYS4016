import math

def exp(mu, lmbda):
    return math.e**(-lmbda/mu)

print(0.6/(exp(20, 60)*0.4 + 0.6))
