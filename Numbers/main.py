#  https://github.com/rhammell/mnist-draw
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from model import SimpleNet
DATA_PATH = '../../data/'
MODEL_PATH = '../models/'
MODEL_NAME = 'pytorch_computer_vision_model.pth'
MODEL_SAVE_PATH = MODEL_PATH + MODEL_NAME


RANDOM_SEED = 863689
np.random.seed(seed=RANDOM_SEED)
torch.manual_seed(seed=RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleNet(input_size=28*28, num_classes=10)

# Load in the saved state_dict()
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
model = model.to(device)

test_data = pd.read_csv('../../data/mnist_test.csv')
class_names = [0,1,2,3,4,5,6,7,8,9]
x_test = test_data.values[:, 1:]
y_test = test_data['label'].values

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
y_preds = []

with torch.inference_mode():
  for X in tqdm(x_test, desc="Making predictions"):
    # Send data and targets to target device
    X = torch.tensor(X.astype('float32')).unsqueeze(0).to(device)
    # Turn predictions from prediction probabilities -> predictions labels
    y_pred = model(X).argmax(-1).cpu()
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred)

y_pred_tensor = torch.cat(y_preds)
y_tensor = torch.tensor(y_test)


confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=y_tensor)


fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
    class_names=class_names,  # turn the row and column labels into class names
    figsize=(10, 7)
);
plt.show()