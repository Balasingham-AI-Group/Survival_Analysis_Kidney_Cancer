#Written by Maryam Mahootiha

#Import Libraries

# In[]:

import time
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import KFold
import random
from torch.utils.data import Subset

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    Activations,
    AsChannelFirstd,
    EnsureChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Spacingd,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityRanged,
    Resized,
    Orientationd,
    ToTensord,
    RandAffined,
    RandGaussianNoised
)

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism


from efficientnet_pytorch_3d import EfficientNet3D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions
import torch.nn as nn
import torch.nn.functional as F

from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.models import CoxPH

from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter


# Get the images,tumors based on the order in ISUP folder

# total files before augmentation

# In[]:


data_dir = '/home/mary/Documents/kidney_ds/ISUP_C'
class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name,'image', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'image'))] 
               for class_name in class_names]

tumor_files = [[os.path.join(data_dir, class_name,'label', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'label'))] 
               for class_name in class_names]

image_file_list = []
tumor_file_list = []
image_label_list = []

for i, class_name in enumerate(class_names):
    
    image_file_list.extend(sorted(image_files[i]))
    tumor_file_list.extend(sorted(tumor_files[i]))
    image_label_list.extend([i] * len(image_files[i]))

    
num_total = len(image_label_list)


print('Total image count:', num_total)
print('Total label count:', len(tumor_file_list))
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])
print("Percent of every class:", [int(((len(image_files[i])/num_total)*100)) for i in range(num_class)])


# See the order of patients and classes

# In[]:


order_of_cases = []
classes = []
for i in image_file_list:
    order_of_cases.append(int(i[58:63]))
    classes.append(int(i[43:44]))


# Making Custom dataset

# In[]:


class KDataset(Dataset):

    def __init__(self, image_files, tumor_files, labels):
        self.image_files = image_files
        self.tumor_files = tumor_files
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        im =  self.image_files[index]['image']
        lb = self.tumor_files[index]['label']
        imlb = torch.cat((im, lb),0)        
        return imlb, self.labels[index]


# Monai transformers

# In[]:


train_transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2)),
    Orientationd(keys=['image'], axcodes="RAS"),
    ScaleIntensityRanged(keys='image', a_min=-200, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['image'], spatial_size=[128, 128, 128]),
    ToTensord(keys=['image'])
])
     
val_transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Orientationd(keys=['image'],axcodes="RAS"),
    ScaleIntensityRanged(keys='image', a_min=-200, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['image'],spatial_size=[128, 128, 128]),
    ToTensord(keys=['image'])
])

lb_transforms = Compose([
    LoadImaged(keys=['label']),
    AddChanneld(keys=['label']),
    Orientationd(keys=['label'],axcodes="RAS"),
    Resized(keys=['label'],spatial_size=[128, 128, 128]),
    ToTensord(keys=['label'])
])


# all files without augmentation

# In[]:


all_files = [{"image": image_name} for image_name in image_file_list]
all_files_l = [{"label": label_name} for label_name in tumor_file_list]
all_im = Dataset(data=all_files, transform=train_transforms)
all_lb = Dataset(data=all_files_l, transform=lb_transforms)


# In[]:


all_ds = KDataset(all_im, all_lb, image_label_list)
all_loader = DataLoader(all_ds, batch_size = 1, shuffle = False)


# Feature Extraction

# In[]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# define model and load model

# In[]:

device = torch.device("cuda:0")
model = EfficientNet3D.from_name("efficientnet-b7", override_params={'num_classes': num_class}, in_channels=2)
model = model.to(device)


# In[]:

model.load_state_dict(torch.load('best_metric_ISUP_fold2.pth'))
model._avg_pooling.register_forward_hook(get_activation('_avg_pooling'))


# get the features of images from model

# In[]:


my_feat = []
for batch_data in all_loader:
    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    outputs = model(inputs)
    my_feat.append(activation['_avg_pooling'])


# flatten the tensor to a vector

# In[]:


flat_tens = []
for i in range(len(my_feat)):
    flati = torch.flatten(my_feat[i])
    flati_cpu = (flati.cpu()).numpy()
    flat_tens.append(flati_cpu)
len(flat_tens)


# save the features to a csv file

# In[]:


x = ''
for i in range(0,2560):
    x+=('feature{},'.format(i))
x1 = x[:len(x)-1]
np.savetxt("244_features_noneaugmented.csv", flat_tens, delimiter=',', header=x1,comments ='')


# Extract survival times and events of patients

# In[]:


def surv_times(Survival_df):

    n_get_target = lambda Survival_df: (Survival_df['case_id'], Survival_df['vital_days_after_surgery'].values.astype(int), Survival_df['event'].values.astype(int))
    raw_target = n_get_target(Survival_df)
    return raw_target

Survival_df = pd.read_csv ('/home/mary/Downloads/kits.csv')
raw_target = surv_times(Survival_df)

numbers_p = []
case_ids_label = []
for i,path in enumerate(image_file_list):
    pathn = path[58:63]
    numbers_p.append(pathn)

case_numbers = []
for i,strnumber in enumerate(numbers_p):
    strnumber = int(strnumber)
    case_numbers.append(strnumber)   


# We have to find survival time and events in the order of files we have based on ISUP classes
# Then we save the case_id, survival, event to a csv file

# In[]:


raw_tar_case = np.array([raw_target[0][i] for i in case_numbers])
raw_tar_st = np.array([raw_target[1][i] for i in case_numbers])
raw_tar_ev = np.array([raw_target[2][i] for i in case_numbers])

dataf = list(zip(raw_tar_case, raw_tar_st, raw_tar_ev))
pp = pd.DataFrame(data = dataf, columns = ['case_id','survival','event'])
pp.to_csv('kits_label_244.csv',index=False)


# Split dataset to train,validation and test

# We use the indexes we had before for image classification in the name of fold0,fold1 and fold2_index.csv. As we used fold 2 for loading model we use the indexes of that fold for train and test. We add the validation to the the training. So the training has 21 dead people and the test has 11 dead people. 

# In[]:


fold2_indexes = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/fold2_indexes.csv')


# In[]:


train_index_fold2 = fold2_indexes['train'].tolist()

val_index_fold2 = fold2_indexes['validation'].tolist()
val_index_fold2 = [x for x in val_index_fold2 if np.isnan(x) == False]
val_index_fold2 = [int(x) for x in val_index_fold2]

test_index_fold2 = fold2_indexes['test'].tolist()
test_index_fold2 = [x for x in test_index_fold2 if np.isnan(x) == False]
test_index_fold2 = [int(x) for x in test_index_fold2]


# merging the vlaidation index with train index

# In[]:


train_val_index_fold2 = sorted(train_index_fold2 + val_index_fold2)


# Survival Analysis

# normalize the feature columns for training model

# In[]:


featuredf = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/244_features_noneaugmented.csv')

df_train = featuredf.loc[train_val_index_fold2]
df_test = featuredf.loc[test_index_fold2]

scaler = StandardScaler()

x_train = scaler.fit_transform(df_train).astype('float32')
x_test = scaler.transform(df_test).astype('float32')


# Extract the targets from kits_label_augmented.csv

# In[]:


label_d = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/kits_label_244.csv')
df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]


# In[]:


num_durations = 15
labtrans = LogisticHazard.label_transform(num_durations)


# Make train and validation dataset

# In[]:


get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))

target_train = labtrans.fit_transform(*get_target(df_train_target))

train = tt.tuplefy(x_train, target_train)


# Extract the died people in train and test dataset
# in died lists, 0 is for indices in train dataframe or test dataframe 1 for survival times and 2 for case ids

# In[]:


caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)


# In[]:


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])


# Define a Logistic hazard net for survival prediction

# In[]:


in_features = x_train.shape[1]
num_nodes = [32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.3

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
new_model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)


# Training

# In[]:


batch_size = 256
epochs = 500
log = model.fit(*train, batch_size, epochs, None, True)


# save model weights

# In[]:


model.save_model_weights('XX.pt')


# Prediction of survival for test dataset after fitting model

# load model weights

# In[]:


model.load_model_weights('XX.pt')


# In[]:


surv_test_disc = model.predict_surv_df(x_test)
surv_train_disc = model.predict_surv_df(x_train)


# In[]:


surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)


# Indexes for evaluating survival model

# C-index
# The Ctd is an extension of Harrell’s concordance index for computing performance of survival models. c-index is between 0 and 1. A Ctd
# of 1 indicates perfect concordance between predicted risk and actual survival, while a value of 0.5 means random concordance

# In[]:

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


# In[]:


len(died_person_test[0])


# Integerated Brier Score
# The IBS is based on the average squared distances between observed survival status and predicted survival probabilities at all available follow up times. It is between 0 and 1. The lower the IBS, the better the model performance, with the best value at zero.

# In[]:


time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

IBS = ev.integrated_brier_score(time_grid)
IBS_train = ev_train.integrated_brier_score(time_grid)

print(f'IBS Score for test: {IBS}')
print(f'IBS_train Score for train: {IBS_train}')


# AUC
# the area under the ROC curve (AUC) can be extended to survival data by defining sensitivity (true positive rate) and specificity (true negative rate) as time-dependent measures. The associated cumulative/dynamic AUC quantifies how well a model can distinguish subjects who fail by a given time, from subjects who fail after this time
# the times is from the minimum time of test and maximum time of test. The points number is round((end_t-start_t)/(model.predict(x_test).squeeze().shape[1]))

# In[]:


import sksurv
from sksurv.metrics import cumulative_dynamic_auc
start_t = min(durations_test)
end_t = max(durations_test)
times = np.arange(start_t, end_t, ((end_t-start_t)/(model.predict(x_test).squeeze().shape[1])))

# start by defining a function to transform 1D arrays to the 
# structured arrays needed by sksurv
def transform_to_struct_array(times, events):
    return sksurv.util.Surv.from_arrays(events, times)

# then call the AUC metric from sksurv
AUC_metric = cumulative_dynamic_auc(
    transform_to_struct_array(durations_train, events_train), 
    transform_to_struct_array(durations_test, events_test),
    model.predict(x_test).squeeze(),
    times)


# A model with an AUC score of 0.5 is no better than a model that performs random guessing
# 0.5 = No discrimination
# 0.5-0.7 = Poor discrimination
# 0.7-0.8 = Acceptable discrimination


# Violin Plot

# In[]:


data_1 = np.asarray(surv_test_disc.iloc[-1, died_person_test[0][:]])
data_2 = np.asarray(surv_test_disc.iloc[-1, alive_person_test[0][:]])
data_3 = np.asarray(surv_train_disc.iloc[-1, died_person_train[0][:]])
data_4 = np.asarray(surv_train_disc.iloc[-1, alive_person_train[0][:]])
data = list([data_1, data_2, data_3, data_4])
fig, ax = plt.subplots()
ax.violinplot(data, showmeans=True, showmedians=False)
ax.set_ylabel('Survival Probabilities')
xticklabels = ['Dead_Test', 'Censored_Test', 'Dead_Train', 'Censored_Train']
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(xticklabels)
ax.yaxis.grid(True)
red = '#B60D0D'
blue = '#23BDC4'
green = '#23C47B'
khaki = '#D2A03D'
violet = '#B86FCD'

violin_parts = ax.violinplot(data, showmeans=True, showmedians=False)

i=0
for vp in violin_parts['bodies']:
    if i==0:
        vp.set_facecolor(blue)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    
    if i ==1:
        vp.set_facecolor(green)
        vp.set_edgecolor(blue)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
        
    if i ==2:
        vp.set_facecolor(violet)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    
    if i ==3:
        vp.set_facecolor(khaki)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    i+=1    
plt.savefig('Violin_plot.eps', format='eps')




# In[]:


kmf = KaplanMeierFitter()
events_test = list(np.ones(len(died_person_test[0])))
kmf.fit(died_person_test[0], events_test) # t = Timepoints, Rx: 0=censored, 1=event
kmf.plot()



# Curves for dead people in test dataset

# In[]:


surv_test_cont.iloc[:, died_person_test[0][:]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


# In[]:


surv_test_cont.iloc[:,53].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


# Curves for censored people in test dataset

# In[]:


surv_test_cont.iloc[:, alive_person_test[0][:]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


# In[]:


kmf = KaplanMeierFitter()
kmf.fit(durations_train, events_train) # t = Timepoints, Rx: 0=censored, 1=event
kmf.plot()


