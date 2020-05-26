import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


csd = pd.read_csv("Q9_a.csv")
print (csd.head())

speed_skew=csd["speed"].skew()
print ("skewness of speed:   ", speed_skew)

dist_skew=csd["dist"].skew()
print ("skewness of dist:   ", dist_skew)

speed_kurtosis=csd["speed"].kurtosis()
print ("kurtosis of speed:   ", speed_kurtosis)

dist_kurtosis=csd["dist"].kurtosis()
print ("kurtosis of dist:   ", dist_kurtosis)


spw= pd.read_csv("Q9_b.csv")
print (spw.head())

sp_skew=spw["SP"].skew()
print ("skewness of SP:   ", sp_skew)

wt_skew=spw["WT"].skew()
print ("skewness of WT:   ", wt_skew)

sp_kurtosis=spw["SP"].kurtosis()
print ("kurtosis of SP:   ", sp_kurtosis)

wt_kurtosis=spw["WT"].kurtosis()
print ("kurtosis of WT:   ", wt_kurtosis)


#visualization1
plt.hist(csd["speed"])
plt.title("Histogram for Speed")
plt.xlabel("speed")
plt.ylabel("Frequency")
plt.show()

plt.hist(csd["dist"])
plt.title("Histogram for dist")
plt.xlabel("dist")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(csd["speed"])
plt.title("Boxplot for Speed")
plt.xlabel("speed")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(csd["dist"])
plt.title("Boxplot for dist")
plt.xlabel("dist")
plt.ylabel("Frequency")
plt.show()



#visualization2
plt.hist(spw["SP"])
plt.title("Histogram for SP")
plt.xlabel("SP")
plt.ylabel("Frequency")
plt.show()

plt.hist(spw["WT"])
plt.title("Histogram for WT")
plt.xlabel("WT")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(spw["SP"])
plt.title("Boxplot for SP")
plt.xlabel("SP")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(spw["WT"])
plt.title("Boxplot for WT")
plt.xlabel("WT")
plt.ylabel("Frequency")
plt.show()
