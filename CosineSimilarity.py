import pandas as pd
from scipy import spatial
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt



df = pd.read_csv("C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\Unwarped\\KURA-UNWARPED_Summary.csv") 

algorithmResults = df['Median Angle'].tolist()

SamplePointResults = df['mla_1'].tolist()

CanopeoResults = df['mla_2'].tolist()

x = []

for i in range(len(algorithmResults)):
    x.append(i)

val_out = 1 - cdist([algorithmResults], [SamplePointResults], 'cosine')

alg_SP_result = 1 - spatial.distance.cosine(algorithmResults, SamplePointResults)

alg_Can_result = 1 - spatial.distance.cosine(algorithmResults, CanopeoResults)

SP_Can_result = 1 - spatial.distance.cosine(SamplePointResults, CanopeoResults)

print(alg_SP_result)

print(SP_Can_result)

print(alg_Can_result)

plt.xlabel("Image Number")
plt.ylabel("Leaf Angle")
plt.title("Results from Proposed Algorithm, mla_1 and mla_2")

for i in range(len(algorithmResults)):
    plt.plot(x[i], algorithmResults[i], marker = 'o', color = 'crimson', label = "Proposed Algorithm" if i == 0 else "")
    plt.plot(x[i], SamplePointResults[i], marker = 's', color = 'black', label = "mla_1" if i == 0 else "")    
    plt.plot(x[i], CanopeoResults[i], marker = 'D', color = 'blue', label = "mla_2" if i == 0 else "")        

plt.legend()
plt.show()
