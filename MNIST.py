import torch
from torchvision import transforms
from fastai.vision.all import *

torch.cuda.get_device_name(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path = untar_data(URLs.MNIST_SAMPLE)

def is_cat(x): return x[0].isupper()


threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255


# Creating training set
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
dset = list(zip(train_x,train_y))


valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))

# Initialises random parameters
def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

# Creating weights and biases
weights = init_params((28*28,1))
bias = init_params(1)

# Defining a model, uses matrix multiplicaiton on the input and weights+biases
def linear1(xb):
    return xb@weights+bias

# Defining loss function
def loss_f(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

# Creating a DataLoader with our dataset split into 256 mini-batches
dl = DataLoader(dset, batch_size=256)

# Creating a validation set DataLoader
valid_dl = DataLoader(valid_dset,batch_size=256)

# Function which makes a prediction, calculates loss and gradient calculations.
def calc_grad(xb,yb,model):
    preds = model(xb)
    loss = loss_f(preds,yb)
    loss.backward()

# Lets define a function which will train for an epoch
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

# define a fucntion which checks the accuracy of a batch
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

# Define a function to return the accuracy of our model on the validation dataset
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

# Lets train our model
lr = 1.
params = weights,bias
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')