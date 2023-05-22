#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


# In[11]:


# Loading training_variants. Its a comma seperated file
data_variants = pd.read_csv('/Users/radhikadalal/Downloads/msk-redefining-cancer-treatment/Dataset/training_variants')
# Loading training_text dataset. This is seperated by ||
data_text = pd.read_csv('/Users/radhikadalal/Downloads/msk-redefining-cancer-treatment/Dataset/training_text', sep='\\|\\|', engine='python', names=['ID', 'TEXT'], skiprows=1)


# FOR THE DATA VARIANTS 

# In[12]:


data_variants.head(3)


# id : link the mutation to the clinical evidence
# Gene : the gene where this genetic mutation is located
# Variation : the aminoacid change for this mutations 
# Class : class value 1-9, this genetic mutation has been classified on

# In[13]:


data_variants.info()


# In[14]:


data_variants.describe()


# In[15]:


data_variants.shape   #checking the shape of the data


# In[16]:


# Checking columns in data set
data_variants.columns


# FOR THE TEXT DATA

# In[17]:


data_text.head(3)


# In[18]:


data_text.info()


# In[19]:


data_text.describe()


# In[20]:


data_text.columns


# In[21]:


# checking the dimentions
data_text.shape


# In[ ]:





# In[22]:


data_variants.Class.unique()


# In[23]:


# removing all stop words like a, is, an, the, ... 
# so we collecting all of them from nltk library


# In[24]:


import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[25]:


def data_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        data_text[col][ind] = string


# In[26]:


for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        data_text_preprocess(row['TEXT'], index, 'TEXT')


#  merging both the dataset by common column ID. merging both gene_variations and text data based on ID

# In[33]:


result = pd.merge(data_variants, data_text,on='ID', how='left')
result.head()


# handling missing values

# In[28]:


result[result.isnull().any(axis=1)]


# In[29]:


result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']


# In[30]:


result[result.isnull().any(axis=1)]


# In[32]:


#check. for blank spaces
y_true = result['Class'].values
result.Gene      = result.Gene.str.replace('\s+', '_')
result.Variation = result.Variation.str.replace('\s+', '_')

splitting data into train, test and validation
# In[34]:


# Splitting the data into train and test set 
X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)
# split the train data now into train validation and cross validation
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)


# In[35]:


print('Number of data points in train data:', train_df.shape[0])
print('Number of data points in test data:', test_df.shape[0])
print('Number of data points in cross validation data:', cv_df.shape[0])


# In[37]:


train_class_distribution = train_df['Class'].value_counts().sort_index()
test_class_distribution = test_df['Class'].value_counts().sort_index()
cv_class_distribution = cv_df['Class'].value_counts().sort_index()


# Visualizing for train class distrubution:

# In[47]:



train_class_distribution.plot(kind='bar', color='red')
plt.xlabel('Class')
plt.ylabel(' Number of Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()


# In[48]:


sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')


# In[49]:


test_class_distribution.plot(kind='bar', color='teal')
plt.xlabel('Class')
plt.ylabel('Number of Data points per Class')
plt.title('Distribution of yi in test data')
plt.grid()
plt.show()


# In[50]:


sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')


# In[52]:


cv_class_distribution.plot(kind='line', color='green')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in cross validation data')
plt.grid()
plt.show()


# In[53]:


sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3), '%)')


# we need log-loss as final evaluation metrics. For doing this we will build a random model and will evaluate log loss. Our model should return lower log loss value than this. 
# we need to generate 9 random numbers because we have 9 class such that their sum must be equal to 1 because sum of Probablity of all 9 classes must be equivalent to 1.

# In[54]:


test_data_len = test_df.shape[0]
cv_data_len = cv_df.shape[0]


# In[55]:


# we create a output array that has exactly same size as the CV data
cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))


# In[56]:


#we create a output array that has exactly same as the test data
test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))


# In[61]:


predicted_y =np.argmax(test_predicted_y, axis=1)


# In[62]:


predicted_y


# In[63]:


predicted_y = predicted_y + 1


# In[64]:


predicted_y


# Confusion Matrix

# In[65]:


C = confusion_matrix(y_test, predicted_y)


# In[70]:


labels = [1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(20,7))
sns.heatmap(C, annot=True, cmap="coolwarm", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# Precision matrix

# In[67]:


B =(C/C.sum(axis=0))


# In[69]:


plt.figure(figsize=(20,7))
sns.heatmap(B, annot=True, cmap="viridis", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# Recall matrix

# In[71]:


A =(((C.T)/(C.sum(axis=1))).T)


# In[72]:


plt.figure(figsize=(20,7))
sns.heatmap(A, annot=True, cmap="jet", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# Evaluating Gene Column

# In[73]:


unique_genes = train_df['Gene'].value_counts()
print('Number of Unique Genes :', unique_genes.shape[0])
# the top 10 genes that occured most
print(unique_genes.head(10))


# In[74]:


s = sum(unique_genes.values);
h = unique_genes.values/s;
c = np.cumsum(h)
plt.plot(c,label='Cumulative distribution of Genes')
plt.grid()
plt.legend()
plt.show()


# we need to convert these categorical variable to appropirate format which my machine learning algorithm will be able to take as an input.
# 
# So we have 2 techniques to deal with it.
# 
# ***One-hot encoding***
# ***Response Encoding*** (Mean imputation)

# In[75]:


# one-hot encoding of Gene feature.
gene_vectorizer = CountVectorizer()
train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])


# Let's check the number of column generated after one hot encoding. One hot encoding will always return higher number of column.

# In[76]:


train_gene_feature_onehotCoding.shape


# In[80]:


gene_onehot = gene_vectorizer.transform(train_df['Gene'])
gene_vectorizer.get_feature_names_out()


# Response encoding columns for Gene column
# 
# 

# In[94]:


def get_gv_fea_dict(alpha, feature, df):
    value_count = train_df[feature].value_counts()
    print(train_df['Gene'].value_counts())


# In[85]:


gv_dict = dict()


# In[ ]:




