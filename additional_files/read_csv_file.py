from pathlib import Path
import pandas as ps
import matplotlib.pyplot as plt



data = ps.read_csv("run_seq_cuda_1000.csv")

#plt.plot(data['losses'])
losses = []
for i in data['losses']:
    if ps.notna(i):
        print(i)
        new = i[7:]
        result = new.split(',')
        print(result[0])
        losses.append(float(result[0]))

plt.plot(losses)
plt.title('Loss for Sequential 1000 iterations')

plt.show()