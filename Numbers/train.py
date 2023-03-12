import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm
from model import SimpleNet
DATA_PATH = '../../data/'
MODEL_PATH = '../models/'
MODEL_NAME = 'pytorch_computer_vision_model.pth'
MODEL_SAVE_PATH = MODEL_PATH + MODEL_NAME


RANDOM_SEED = 863689
np.random.seed(seed=RANDOM_SEED)
torch.manual_seed(seed=RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class MNISTDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.X = torch.tensor(data_x.astype('float32'))
        self.y = torch.tensor(data_y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_data = pd.read_csv('../../data/mnist_train.csv')
test_data = pd.read_csv('../../data/mnist_test.csv')

x_train = train_data.values[:, 1:]
y_train = train_data['label'].values
x_test = test_data.values[:, 1:]
y_test = test_data['label'].values


train_dataset = MNISTDataset(x_train, y_train)
test_dataset = MNISTDataset(x_test, y_test)
batch_size = 1024
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = SimpleNet(input_size=28*28, num_classes=10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(model, criterion, optimizer,
          train_dataloader, test_dataloader, num_epochs):

    train_accuracy_ = np.zeros(num_epochs)
    test_accuracy_ = np.zeros(num_epochs)

    for i_epoch in tqdm(range(num_epochs)):

        train_accuracy = 0
        test_accuracy = 0

        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            y_ = model(X)

            optimizer.zero_grad()

            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            train_accuracy += (y_.argmax(-1).detach() == y).cpu().numpy().mean()

        train_accuracy /= len(train_dataloader)
        train_accuracy_[i_epoch] = train_accuracy

        model.eval()
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            with torch.inference_mode():
                y_ = model(X)

                # save loss and accuracy
                test_accuracy += (y_.argmax(-1) == y).cpu().numpy().mean()

        test_accuracy /= len(test_dataloader)

        test_accuracy_[i_epoch] = test_accuracy

    return train_accuracy_, test_accuracy_

epochs = 20
train_accuracy_, test_accuracy_ = train(model, criterion=criterion,
                          optimizer=optimizer,
                          train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          num_epochs=epochs)

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(np.arange(train_accuracy_.shape[0]), train_accuracy_,
           label='train_accuracy')
ax.plot(np.arange(test_accuracy_.shape[0]), test_accuracy_,
           label='test_accuracy')
ax.set_xticks(np.arange(0, train_accuracy_.shape[0], 4))
ax.set_xlabel('num_epoch')
ax.set_ylabel('accuracy')
ax.legend()
ax.grid()
plt.show()

fig, axes = plt.subplots(3, 5, figsize=(13, 9))
model.eval()
for i in range(3):
    for j in range(5):
        index = np.random.choice(x_test.shape[0])
        axes[i, j].imshow(x_test[index].reshape(28, 28), )
        with torch.inference_mode():
            input_data = x_test[index].astype('float32')
            model_input = torch.tensor(input_data).unsqueeze(0).to(device)
            pred = model(model_input).argmax(-1).cpu().numpy()
        axes[i, j].set_title(f'Predicted label: {pred}')
        axes[i, j].axis('off')
fig.tight_layout()
plt.show()

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)