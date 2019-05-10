
# coding: utf-8

# ### NN implementation

# In[2]:


# Helpers fcts.
def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_multiclass_loss(y, y_fwrd):
    l_sum = np.sum(np.multiply(y, np.log(y_fwrd)))
    m = y.shape[1]
    loss = -(1/m) * l_sum
    return loss

def precision(y, y_fwrd):
    n = y.shape[1]
    return np.sum([np.argmax(y_fwrd[:,i]) == np.argmax(y[:,i]) for i in range(n)]) / n



class NeuralNetwork:
    """ 
        Standard, feed forward NN's implementation
    """
    def __init__(self, X, y, n_h):
        self.n_x = X.shape[0]
        self.w1 = np.random.randn(n_h, self.n_x)
        self.b1 = np.zeros((n_h, 1))
        self.n_y = y.shape[0]
        self.w2 = np.random.randn(self.n_y, n_h)
        self.b2 = np.zeros((self.n_y, 1))    

    def train(self, X, y, learn_rate=1):
        
        # feed forward
        z1 = np.matmul(self.w1, X) + self.b1
        ff1 = sigmoid(z1) # feed forward layer 1
        z2 = np.matmul(self.w2, ff1) + self.b2
        ff2 = np.exp(z2) / np.sum(np.exp(z2), axis=0) # feed forward layer 2
        
        # backprop: relations are found by variational calculus on the optimisation function
        m = y.shape[1]
        dz2 = ff2 - y
        dw2 = (1./m) * np.matmul(dz2, ff1.T)
        db2 = (1./m) * np.sum(dz2, axis=1, keepdims=True)

        dff1 = np.matmul(self.w2.T, dz2)
        dz1 = dff1 * sigmoid(z1) * (1 - sigmoid(z1))
        dw1 = (1./m) * np.matmul(dz1, X.T)
        db1 = (1./m) * np.sum(dz1, axis=1, keepdims=True)

        self.w2 = self.w2 - learn_rate * dw2
        self.b2 = self.b2 - learn_rate * db2
        self.w1 = self.w1 - learn_rate * dw1
        self.b1 = self.b1 - learn_rate * db1
        
        loss = compute_multiclass_loss(y, ff2)
        prec = precision(y, ff2)
        
        return {'loss': loss, 'prec': prec}
    
    def test(self, X, y):
        
        z1 = np.matmul(self.w1, X) + self.b1
        ff1 = sigmoid(z1)
        z2 = np.matmul(self.w2, ff1) + self.b2
        ff2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)
        
        loss = compute_multiclass_loss(y, ff2)
        prec = precision(y, ff2)
        
        return {'loss': loss, 'prec': prec}


# ### MNIST Dataset

# In[21]:


# Import
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

# Preprocessing
X = X / 255
digits = 10
examples = y.shape[0]

# hot encoding a la python
y = y.reshape(1, examples)
y = np.eye(digits)[y.astype('int32')]
y = y.T.reshape(digits, examples)

# Build train/test datasets
train_frac = .8 
m_train = int(X.shape[0]*train_frac)
X_train, X_test = X[:m_train].T, X[m_train:].T
y_train, y_test = y[:,:m_train], y[:,m_train:]

# it is standard to seek for balance in updates for NNs.
shuffle_index = np.random.permutation(m_train)
X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index];


# In[3]:


n_h = 64
nn = NeuralNetwork(X_train, y_train, n_h)

loss_conv, prec_conv = [], []
for i in tqdm(range(2500)):
    scores = nn.train(X_train, y_train)
    if (i % 10 == 0):
        print("Epoch", i, "loss: ", scores['loss'])
        print("Epoch", i, "precision: ", scores['prec'])
        loss_conv.append(scores['loss'])
        prec_conv.append(scores['prec'])


# In[4]:


scores = nn.test(X_test, y_test)
print("Epoch", i, "loss: ", scores['loss'])
print("Epoch", i, "precision: ", scores['prec'])


# #### With one hidden layer, we observed normal behaviour for training and already good scores for classificatio on the test dataset! Probably the neural network could have been trained longer before overfitting.This hypothesis could have been tested in comparing classification scores on the test dataset after various number of epochs. 
# 
# #### However, because we train on the full dataset each time, we could not do that. The standard improvement would have been to have a train method decomposing the data set in many batches for a lower learning rate...
# 
# #### Similarly, the above code could be refactored to allow for arbitrary number of layers.

# In[20]:


plt.subplot(2, 1, 1)
plt.title('Precision and Loss during training')
plt.plot(prec_conv)
plt.ylabel('Precision')
plt.subplot(2, 1, 2)
plt.plot(loss_conv)
plt.ylabel('Loss');


# ### Spam Base Dataset 

# In[5]:


# https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/
data = []
f = open('data/spambase.data', "r")
data = f.readlines()
np.random.shuffle(data)
f.close()

X = np.array(list(map(lambda d: [float(x) for x in d[:-1].split(',')[:-1]], data)))
to_class = lambda x: np.array([1, 0]) if x==0 else np.array([0, 1])
y = np.array(list(map(lambda d: to_class(int(d[:-1].split(',')[-1])), data)))
n_data, train_frac = X.shape[0], 0.8

# massaging data into compatible formats as:
m = int(n_data*train_frac)
X_train, y_train = X[:m].T, y[:m].T
X_test, y_test = X[m:].T, y[m:].T


# In[8]:


n_h = 256 # 128
nn = NeuralNetwork(X_train, y_train, n_h)

loss_conv_2, prec_conv_2 = [], []
for i in tqdm(range(2500)):
    scores = nn.train(X_train, y_train)
    if (i % 10 == 0):
        print("Epoch", i, "loss: ", scores['loss'])
        print("Epoch", i, "precision: ", scores['prec'])
        loss_conv_2.append(scores['loss'])
        prec_conv_2.append(scores['prec'])


# In[9]:


scores = nn.test(X_test, y_test)
print("Epoch", i, "loss: ", scores['loss'])
print("Epoch", i, "precision: ", scores['prec'])


# #### Below, during training, NN's seem to be oscillating between local minima. Training with batches of the total data should improve this behaviour.

# In[18]:


plt.subplot(2, 1, 1)
plt.title('Precision and Loss during training')
plt.plot(prec_conv_2)
plt.ylabel('Precision')
plt.subplot(2, 1, 2)
plt.plot(loss_conv_2)
plt.ylabel('Loss');

