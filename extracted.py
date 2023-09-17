import torch
from torchvision import transforms
from fastai.vision.all import *

torch.cuda.get_device_name(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path = untar_data(URLs.PETS) / 'images'

def is_cat(x): return x[0].isupper()


dogsAndCats = get_image_files(path)

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

# Lets declare our lR to be 1 and initialise params.
lr = 1.
params = weights,bias

# Okay this didn't work lets try something new.
# Recall our learning function `linear1` and our param initialiser `init_params`
# We can replace this with the following

linear_model = nn.Linear((192*192)*3,1)

# Lets define a basic optimiser

class BasicOptim:
    def __init__(self,params,lr):
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs): # Take step, adjusting params for gradients and learning rate
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: # Reset gradients of params after each step
            p.grad = None

# Lets create the optimiser now
opt = BasicOptim(linear_model.parameters(),lr)

# Earlier we defined `train_epoch`, which would update params, but now this is handled by `opt` so lets simplify it
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


# We can now define `train_model` as follows
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')

# However we can still make this simpler
# our `basicOptim` can be entirely replaced by the `SGD` class in PyTorch
# Thus we can fully define with the following.

linear_model = nn.Linear((192*192)*3,1)
opt = SGD(linear_model.parameters(), lr)

# train_model is also redundant, fastAI provides learner.fit
# We're gonna need a dataloader
dls = DataLoaders(dl,valid_dl)

# Lets now add a non-linearity
complex_net = nn.Sequential(
    nn.Linear((192*192)*3, 64),  # Adding a linear layer with 64 units
    nn.ReLU(),
    nn.Linear(64, 64),           # Another one with 64 units
    nn.ReLU(),
    nn.Linear(64, 32),           # And another with 32 units
    nn.ReLU(),
    nn.Linear(32, 1)             # Finally, your output layer
)
learn = Learner(dls, complex_net, opt_func=SGD,
                loss_func=loss_f, metrics=batch_accuracy)
learn.fit(20,1)