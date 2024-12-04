import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from models import AllCNN
from models import ResNet18
from datasets import AgeDB
from unlearn import *
from metrics import *
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
# from IPython import get_ipython
import seaborn as sns
import nltk
# nltk.download('stopwords')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Load your DataFrame from the CSV file
df = pd.read_csv("../agedb.csv")
# Define the data split percentages
train_ratio = 0.5
val_ratio = 0.1
test_ratio = 0.4

# Calculate the number of rows for each split
n = len(df)
num_train = int(train_ratio * n)
num_val = int(val_ratio * n)

# Create 'split' values for each split
split_values = ['train'] * num_train + ['val'] * num_val + ['test'] * (n - num_train - num_val)

# Shuffle the split values
np.random.shuffle(split_values)

# Assign the 'split' values to the DataFrame
print('Loading the Data: \n')
df['split'] = split_values
df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
train_data = AgeDB(data_dir='', df=df_train, img_size=32, split='train')
val_data = AgeDB(data_dir='', df=df_val, img_size=32, split='val')
test_data = AgeDB(data_dir='', df=df_test, img_size=32, split='test')

train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=False)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ## Training a model on all data
# device = 'cpu'
print('Training Main Model:\n', '-'*70)
#model = AllCNN(num_classes=1).to(device)
model= ResNet18().to(device)
epochs = 100
save_path = "../cache/AllCNN_AgeDB_100epochs.pt"
history = fit_one_cycle(epochs, model, train_loader, val_loader, device = device, save_path = save_path)
print("Loading Main trained model:\n", '-'*70)
model.load_state_dict(torch.load(save_path))


# ## Creating separate forget and retain sets from data
print('Creating separate forget and retain sets from data: \n', '-'*70)
df_train_forget = df_train[df_train['age'] <= 30]
df_val_forget = df_val[df_val['age'] <= 30]
df_test_forget = df_test[df_test['age'] <= 30]
train_data_forget = AgeDB(data_dir='', df=df_train_forget, img_size=32, split='train')
val_data_forget = AgeDB(data_dir='', df=df_val_forget, img_size=32, split='val')
test_data_forget = AgeDB(data_dir='', df=df_test_forget, img_size=32, split='test')
train_forget_loader = DataLoader(train_data_forget, batch_size=256, shuffle=False,
                              num_workers=64, pin_memory=True, drop_last=False)
val_forget_loader = DataLoader(val_data_forget, batch_size=256, shuffle=False,
                            num_workers=64, pin_memory=True, drop_last=False)
test_forget_loader = DataLoader(test_data_forget, batch_size=256, shuffle=False,
                             num_workers=64, pin_memory=True, drop_last=False)
df_train_retain = df_train[df_train['age'] > 30]
df_val_retain = df_val[df_val['age'] > 30]
df_test_retain = df_test[df_test['age'] > 30]

train_data_retain = AgeDB(data_dir='', df=df_train_retain, img_size=32, split='train')
val_data_retain = AgeDB(data_dir='', df=df_val_retain, img_size=32, split='val')
test_data_retain = AgeDB(data_dir='', df=df_test_retain, img_size=32, split='test')

train_retain_loader = DataLoader(train_data_retain, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)
val_retain_loader = DataLoader(val_data_retain, batch_size=256, shuffle=False,
                            num_workers=64, pin_memory=True, drop_last=False)
test_retain_loader = DataLoader(test_data_retain, batch_size=256, shuffle=False,
                             num_workers=64, pin_memory=True, drop_last=False)

main_retain_eval = evaluate(model, test_retain_loader, device=device)
print(f'Evaluating Main Model on Retained Data = {main_retain_eval}')

main_forget_eval = evaluate(model, test_forget_loader, 'cuda')
print(f'Evaluating Main Model on Forget Data = {main_forget_eval}')

#Train only on Retain data(GOLD MODEL/RETRAINED MODEL/NAIVE MODEL)
# device = 'cuda'
print('Train only on Retain data(GOLD MODEL/RETRAINED MODEL/NAIVE MODEL):\n', '-'*70)
#gold_model = AllCNN(num_classes=1).to(device)
gold_model = ResNet18().to(device)
epochs = 100
save_path = "../cache/AllCNN_AgeDB_100epochs_0to30_Gold.pt"
history = fit_one_cycle(epochs, gold_model, train_retain_loader, val_retain_loader, device = device, save_path = save_path)
print("Loading Exact Unlearned trained model:\n", '-'*70)
gold_model.load_state_dict(torch.load(save_path))

gold_retain_eval = evaluate(gold_model, test_retain_loader, device = device)
print(f'Evaluating Exact Unlearned Model on Retained Data = {gold_retain_eval}')

gold_forget_eval = evaluate(gold_model, test_forget_loader, device = device)
print(f'Evaluating Exact Unlearned Model on Forget Data = {gold_forget_eval}')



# ## Unlearning using different methods

# ### Gaussian-Amnesiac
print('Unlearning using Gaussian-Amnesiac Method: \n', '-'*70)
mean = df_train['age'].mean()
sd = df_train['age'].std()
random_preds = np.random.normal(loc=mean, scale=sd, size=(len(df_train[df_train['age'] <= 30]),))
amnesiac_finetune_df = df_train.copy()
amnesiac_finetune_df.loc[amnesiac_finetune_df['age'] <=30, 'age'] = random_preds
amnesiac_finetune_train_data = AgeDB(data_dir='./data', df=amnesiac_finetune_df, img_size=32, split='train')
amnesiac_finetune_train_loader = DataLoader(amnesiac_finetune_train_data, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)
#obtaining unlearned model "amn_model"
#amn_model = AllCNN(num_classes=1).to(device)
amn_model = ResNet18().to(device)
amn_model.load_state_dict(torch.load("/home/rajdeep/Codes/Machine_Unlearning/Blindspot_model/cache/AllCNN_AgeDB_100epochs.pt"))
epochs = 1
save_path = "../cache/AllCNN_AgeDB_1epoch_0to30_Amnesiac_Finetune_Forget_tmp.pt"
history = fit_one_cycle(epochs, amn_model, amnesiac_finetune_train_loader, val_retain_loader, lr = 0.001, device = device, save_path = save_path)


amn_retain_eval = evaluate(amn_model, test_retain_loader, device = device)
print(f'Evaluating Gaussian-Amnesiac Model on Retained Data = {amn_retain_eval}')

amn_forget_eval = evaluate(amn_model, test_forget_loader, device = device)
print(f'Evaluating Gaussian-Amnesiac Model on Forget Data = {amn_forget_eval}')


# ### Blindspot Unlearning
print('\n\nUnlearning using Blindspot Unlearning Method: \n', '+'*70)
df_train_forget['unlearn'] = 1
df_val_forget['unlearn'] = 1
df_test_forget['unlearn'] = 1
df_train_retain['unlearn'] = 0
df_val_retain['unlearn'] = 0
df_test_retain['unlearn'] = 0

udf_train = pd.concat([df_train_forget, df_train_retain])
utrain_data = UAgeDB(data_dir='./data', df=udf_train, img_size=32, split='train')
utrain_loader = DataLoader(utrain_data, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)

# Making the blindspot model
epochs = 2
print(f'Training Blindspot Model(B) for {epochs} epochs: \n', '-'*70)
#bdst_model = AllCNN(num_classes=1).to(device)
bdst_model = ResNet18().to(device)
save_path = "../cache/AllCNN_AgeDB_2epochs_0to30_Proxy_tmp.pt"
history = fit_one_cycle(epochs, bdst_model, train_retain_loader, val_retain_loader, device = device, save_path = save_path)


#Obtaining the unlearned model
print('Updating the original pretrained model on the whole dataset:\n')
#bdstu_model = AllCNN(num_classes=1).to(device)
bdstu_model = ResNet18().to(device)
bdstu_model.load_state_dict(torch.load("../cache/AllCNN_AgeDB_100epochs.pt"))
epochs = 1
save_path = "../cache/AllCNN_AgeDB_0to30_1epochs_ATBeta_50_unlearn_tmp.pt"
history = fit_one_forget_cycle(epochs, bdstu_model, bdst_model,  utrain_loader, val_loader, lr = 0.001, device = device, save_path = save_path)


eval_bdstu_retain = evaluate(bdstu_model, test_retain_loader, device = device)
print(f'Blindspot Model on Unlearned model on the Test Retained Set = {eval_bdstu_retain}')

eval_bdstu_forget = evaluate(bdstu_model, test_forget_loader, device = device)
print(f'Blindspot Model on Unlearned model on the Test Forget Set = {eval_bdstu_forget}')


# ## Comparing the models 

# ### Wasserstein Distance
print('Test:\n')
print('Running the Exact Unleared model on the train forget dataset: \n')
gold_predict = predict(gold_model, train_forget_loader, device = device)
gold_outputs = torch.squeeze(gold_predict).cpu().numpy()

#Original Model
print('Running the Original model(to be unlearned) on the train forget dataset: \n')
full_predict = predict(model, train_forget_loader, device = device)
full_outputs = torch.squeeze(full_predict).cpu().numpy()

wh_dist_full_gold = wasserstein_distance(full_outputs, gold_outputs)
print(f'W-H Distance between Original model and Exact Unleared model on the train forget dataset = {wh_dist_full_gold} \n')

#Gaussian Amnesiac Model
print('Running the Gaussian-Amnesiac Model on the train forget dataset: \n')
amn_predict = predict(amn_model, train_forget_loader, device =device)
amn_outputs = torch.squeeze(amn_predict).cpu().numpy()
wh_dist_amn_gold = wasserstein_distance(amn_outputs, gold_outputs)
print(f'W_H Distance between Gaussian-Amnesiac-Unlearned model and Exact Unlearned Model = {wh_dist_amn_gold}')

#Blindspot Unlearned Model
print('Running the Blindspot Model on the train forget dataset: \n')
bdstu_predict = predict(bdstu_model, train_forget_loader, device =device)
bdstu_outputs = torch.squeeze(bdstu_predict).cpu().numpy()
wh_dist_bdstu_gold = wasserstein_distance(bdstu_outputs, gold_outputs)
print(f'W_H Distance between Blindspot-Unlearned model and Exact Unlearned Model = {wh_dist_bdstu_gold}')

# ### Membership Attack Probabilities

# sample_size = 2000
# att_train_data = AgeDB(data_dir='./data', df=df_train.sample(sample_size), img_size=32, split='train')
# att_val_data = AgeDB(data_dir='./data', df=df_val.sample(sample_size), img_size=32, split='val')
# att_test_data = AgeDB(data_dir='./data', df=df_test.sample(sample_size), img_size=32, split='test')
# att_train_loader = DataLoader(att_train_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)
# att_val_loader = DataLoader(att_val_data, batch_size=256, shuffle=False,
#                             num_workers=10, pin_memory=True, drop_last=False)
# att_test_loader = DataLoader(att_test_data, batch_size=256, shuffle=False,
#                              num_workers=10, pin_memory=True, drop_last=False)

# att_retain_data = AgeDB(data_dir='./data', df=df_train[df_train['age'] > 30].sample(sample_size), img_size=32, split='train')
# att_forget_data = AgeDB(data_dir='./data', df=df_train[df_train['age'] <= 30].sample(sample_size), img_size=32, split='train')
# att_forget_test_data = AgeDB(data_dir='./data', df=df_test[df_test['age'] <= 30].sample(min(sample_size, len(df_test[df_test['age'] <= 30]))), img_size=32, split='test')


# att_forget_loader = DataLoader(att_forget_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)
# att_forget_test_loader = DataLoader(att_forget_test_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)
# att_retain_loader = DataLoader(att_retain_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)


# prediction_loaders = {"forget_data":att_forget_loader}


# #Original Model
# get_membership_attack_prob(att_train_loader, att_test_loader, model, prediction_loaders)


# #Retrained Model
# get_membership_attack_prob(att_train_loader, att_test_loader, gold_model, prediction_loaders)



#Gaussian Amnesiac Model
# get_membership_attack_prob(att_train_loader, att_test_loader, amn_model, prediction_loaders)


# #Blindspot Unlearned Model
# get_membership_attack_prob(att_train_loader, att_test_loader, bdstu_model, prediction_loaders)


# # ### AIN

# #Gaussian Amnesiac Model
# ain(model, amn_model, gold_model, train_data, val_data_retain, val_data_forget, 
#                   batch_size = 256, error_range = 0.05, lr = 0.01, device = device)

# #Blindspot Unlearned Model
# ain(model, bdstu_model, gold_model, train_data, val_data_retain, val_data_forget, 
#                   batch_size = 256, error_range = 0.05, lr = 0.01, device = device)


# # ### Distribution Comparison

# labels = df_train[df_train['age'] <= 30]['age'].values


# pred_norm_dict = {'original model':abs(full_outputs-gold_outputs)/labels,
#             'ours: blindspot':abs(bdstu_outputs-gold_outputs)/labels,
#             'g-amnesiac':abs(amn_outputs-gold_outputs)/labels}

# pred_norm_df = pd.DataFrame(pred_norm_dict)


# plt.rcParams['font.size'] = '14'

# sns.set_style("darkgrid")
# sns.histplot(pred_norm_df, element="poly", stat='density')
# plt.xlabel("Relative prediction difference from retrained model")
# plt.xlim(0,2)




# sample_size = 2000
# att_train_data = AgeDB(data_dir='./data', df=df_train.sample(sample_size), img_size=32, split='train')
# att_val_data = AgeDB(data_dir='./data', df=df_val.sample(sample_size), img_size=32, split='val')
# att_test_data = AgeDB(data_dir='./data', df=df_test.sample(sample_size), img_size=32, split='test')
# att_train_loader = DataLoader(att_train_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)
# att_val_loader = DataLoader(att_val_data, batch_size=256, shuffle=False,
#                             num_workers=10, pin_memory=True, drop_last=False)
# att_test_loader = DataLoader(att_test_data, batch_size=256, shuffle=False,
#                              num_workers=10, pin_memory=True, drop_last=False)

# att_retain_data = AgeDB(data_dir='./data', df=df_train[df_train['age'] > 30].sample(sample_size), img_size=32, split='train')
# att_forget_data = AgeDB(data_dir='./data', df=df_train[df_train['age'] <= 30].sample(sample_size), img_size=32, split='train')
# att_forget_test_data = AgeDB(data_dir='./data', df=df_test[df_test['age'] <= 30].sample(min(sample_size, len(df_test[df_test['age'] <= 30]))), img_size=32, split='test')


# att_forget_loader = DataLoader(att_forget_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)
# att_forget_test_loader = DataLoader(att_forget_test_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)
# att_retain_loader = DataLoader(att_retain_data, batch_size=256, shuffle=True,
#                               num_workers=10, pin_memory=True, drop_last=False)


# prediction_loaders = {"forget_data":att_forget_loader}


# #Original Model
# get_membership_attack_prob(att_train_loader, att_test_loader, model, prediction_loaders)


# #Retrained Model
# get_membership_attack_prob(att_train_loader, att_test_loader, gold_model, prediction_loaders)



# #Gaussian Amnesiac Model
# get_membership_attack_prob(att_train_loader, att_test_loader, amn_model, prediction_loaders)


# #Blindspot Unlearned Model
# get_membership_attack_prob(att_train_loader, att_test_loader, bdstu_model, prediction_loaders)


# # ### AIN

# #Gaussian Amnesiac Model
# ain(model, amn_model, gold_model, train_data, val_data_retain, val_data_forget, 
#                   batch_size = 256, error_range = 0.05, lr = 0.01, device = device)

# #Blindspot Unlearned Model
# ain(model, bdstu_model, gold_model, train_data, val_data_retain, val_data_forget, 
#                   batch_size = 256, error_range = 0.05, lr = 0.01, device = device)

