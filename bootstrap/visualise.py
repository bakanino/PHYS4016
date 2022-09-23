import numpy as np
import matplotlib.pyplot as plt

lsat = np.array([576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594])
gpa = np.array([3.3900, 3.3000, 2.8100, 3.0300, 3.4400, 3.0700, 3.0000, 3.4300, 3.3600, 3.1300, 3.1200, 2.7400, 2.7600, 2.8800, 2.9600])

plt.scatter(lsat, gpa, marker='+')
plt.xlabel('LSAT score')
plt.ylabel('Final GPA')
plt.savefig('plot.png')
