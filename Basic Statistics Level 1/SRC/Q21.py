import pandas as pd

data=pd.read_csv("Cars.csv")
print (data.head())
data_new=data.drop(["HP", "VOL", "SP", "WT"], axis=1)
print (data_new.head())

import matplotlib.pyplot as plt
plt.hist(data["MPG"])
plt.xlabel("MPG")
plt.ylabel("Frequency")
plt.show()


data1=pd.read_csv("wc-at.csv")
plt.hist(data1["Waist"])
plt.xlabel("Waist")
plt.ylabel("Frequency")
plt.show()

plt.hist(data1["AT"])
plt.xlabel("AT")
plt.ylabel("Frequency")
plt.show()
