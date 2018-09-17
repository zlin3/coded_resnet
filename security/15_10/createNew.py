import numpy as np
import sys

n = int(sys.argv[1])

p = []
for i in range(1, n+1):
#for i in range(n, 0, -1):
    p.append(np.loadtxt('./prediction_values/prediction_%d.txt' % i))

p = np.array(p)
p = np.transpose(p)

np.savetxt('parities_new.txt', p, fmt='%i')

s = []
for i in range(1, n+1):
#for i in range(n, 0, -1):
    s.append(np.loadtxt('./softmax_values/softmax_%d.txt' % i)[:, 1])

s = np.array(s)
s = np.transpose(s)

np.savetxt('softmaxes_new.txt', s, fmt='%f')

