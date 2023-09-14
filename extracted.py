import torch
from torchvision import transforms
from fastai.vision.all import *

torch.cuda.get_device_name(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path = untar_data(URLs.PETS) / 'images'

def is_cat(x): return x[0].isupper()


dogsAndCats = (path).ls().sorted()

validation_set_percentage = 0.20
num_of_validation = int(len(dogsAndCats) * validation_set_percentage)
random.seed(42)
random.shuffle(dogsAndCats)

validation_set = dogsAndCats[:num_of_validation]
training_set = dogsAndCats[num_of_validation:]


transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor()
])

# Load and convert all images to RGB mode
training_tensor = [transform(Image.open(o).convert('RGB')) for o in training_set]
validation_tensor = [transform(Image.open(o).convert('RGB')) for o in validation_set]
print(len(training_tensor), len(validation_tensor))

# Creating training set
stacked_images = torch.stack(training_tensor).float() / 255
train_x = stacked_images.view(-1,(192*192)*3)

labels = []

for file_path in training_set:
    filename = file_path.name
    label = 0 if is_cat(filename[0]) else 1  # is_cat was defined earlier
    labels.append(label)

train_y = torch.tensor(labels).unsqueeze(1)
train_y.shape

dset = list(zip(train_x, train_y))

# validation set
stacked_validation_images = torch.stack(validation_tensor).float() / 255
valid_x = stacked_validation_images.view(-1,(192*192)*3)

valid_labels = []

for file_path in validation_set:
    filename = file_path.name
    label = 0 if is_cat(filename[0]) else 1
    valid_labels.append(label)


valid_y = torch.tensor(valid_labels).unsqueeze(1)

valid_dset = list(zip(valid_x, valid_y))

# Initialises random parameters
def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

# Creating weights and biases
weights = init_params((192*192)*3,1)
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