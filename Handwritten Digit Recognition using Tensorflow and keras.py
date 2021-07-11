#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)


# In[4]:


len(X_test)


# In[5]:


X_train[0].shape


# In[6]:


X_train[0]


# In[7]:


plt.matshow(X_train[0])


# In[8]:


y_train[2]


# In[9]:


y_train[:5]


# In[10]:


X_train.shape


# In[11]:


X_train = X_train / 255


# In[12]:


X_train_flattened = X_train.reshape(len(X_train),28 * 28)
X_train_flattened.shape


# In[13]:


X_test.shape


# In[14]:


X_test = X_test / 255


# In[15]:


X_train[0]


# In[16]:


X_test_flattened = X_test.reshape(len(X_test), 28 * 28)
X_test_flattened.shape


# In[17]:


X_train_flattened[0]


# In[18]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784, ), activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train_flattened, y_train, epochs = 10)


# In[19]:


model.evaluate(X_test_flattened, y_test)


# In[20]:


plt.matshow(X_test[1])


# In[21]:


y_predicted = model.predict(X_test_flattened)
y_predicted[1]


# In[22]:


np.argmax(y_predicted[1])


# In[23]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[24]:


y_test[:5]


# In[25]:


cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
cm


# In[26]:


import seaborn as sn 
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[27]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784, ), activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train_flattened, y_train, epochs = 5)


# In[28]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784, ), activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train_flattened, y_train, epochs = 10)


# In[29]:


model.evaluate(X_test_flattened, y_test)


# In[30]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
cm


# In[31]:


import seaborn as sn 
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




