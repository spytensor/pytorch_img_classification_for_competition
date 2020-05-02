import pandas as pd 
import numpy as np 
import os
from IPython import embed

file1 = pd.read_csv("./csvs/efficientnet-b3-model_512-_adam_aug_confidence.csv",header=None)
file2 = pd.read_csv("./csvs/efficientnet-b5-model_456_ranger_aug_confidence.csv",header=None)
file3 = pd.read_csv("./csvs/efficientnet-b4-model_380_ranger_aug_confidence.csv",header=None)

filenames,labels = [],[]
# embed()
# for (filename1,label1),(filename2,label2),(filename3,label3),(filename4,label4),(filename5,label5) in zip(file1.values,file2.values,file3.values,file4.values,file5.values):
for (filename1,label1) ,(filename2,label2),(filename3,label3) in zip(file1.values,file2.values,file3.values):
    filename = filename1
    filenames.append(filename)
    #embed()
    label1 = np.array(list(map(float,label1.split("-"))))
    label2 = np.array(list(map(float,label2.split("-"))))
    label3 = np.array(list(map(float,label3.split("-"))))
    # label4 = np.array(list(map(float,label4.split("[")[1].split("]")[0].split(","))))
    # label5 = np.array(list(map(float,label5.split("[")[1].split("]")[0].split(","))))
    label = np.argmax((label1 + label2 + label3) / 3.0) + 1
    labels.append(label)

submission = pd.DataFrame({'FileName': filenames, 'type': labels})
submission.to_csv("./ensemble_efficientnets.csv", header=None, index=False)

