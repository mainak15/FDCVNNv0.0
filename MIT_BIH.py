import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
import torch.nn.functional as F
import cv2
from scipy import fftpack
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

early_stopping_patience=10

lr = 0.01


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

        
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = np.inf
    patience_counter = 0
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):

        print(f'Present Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(torch.abs(outputs), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(torch.abs(outputs), 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(torch.abs(outputs), labels)

                val_loss += loss.item()
                _, predicted = torch.max(torch.abs(outputs), 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

        
        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './Saved_Model/MIT_log_1506RB4.pth')
            print(f"Model weights saved. Best validation loss: {best_val_loss} (epoch {epoch+1})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Validation loss did not improve for {early_stopping_patience} epochs. Early stopping...')
            break

    
    model.load_state_dict(torch.load('./Saved_Model/MIT_log_1506RB4.pth')) 
    
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(torch.abs(outputs), 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    
    class_names = [str(i) for i in range(5)]  
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(all_labels, all_preds, class_names)

    return train_losses, val_losses, train_accuracies, val_accuracies

class FrequencyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FrequencyConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if in_channels != out_channels:
            self.filters = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.complex64))
        else:
            self.filters = nn.Parameter(torch.randn(out_channels, kernel_size, kernel_size, dtype=torch.complex64))
        self.Initialization()

    def Initialization(self):

        bound = 1 / (self.in_channels ** 0.5)  
        nn.init.uniform_(self.filters.real, -bound, bound)
        nn.init.uniform_(self.filters.imag, -bound, bound)        



    def forward(self, inputs):
        batch_size, in_channels, height, width = inputs.shape
        if self.in_channels != self.out_channels:
            inputs_expanded = inputs.unsqueeze(1)
            filters_expanded = self.filters.unsqueeze(0)
            freq_output = inputs_expanded * filters_expanded
            freq_output = freq_output.sum(dim=2)
        else:
            freq_output = inputs * self.filters
        return freq_output

'''def 𝔣ReLU(input_complex):
    real = torch.relu(input_complex.real)
    imag = torch.relu(input_complex.imag)
    return torch.complex(real, imag)'''

def Log_Magnitude(input_complex):
    mag = torch.abs(input_complex)
    mag_transformed = torch.log1p(mag)
    phase = torch.angle(input_complex)
    output = mag_transformed * torch.exp(1j * phase)
    return output


'''def Cardioid(input_complex):

    return 0.5*(1 + torch.cos(torch.angle(input_complex)))*input_complex'''


class FrequencyInstanceNorm2D(nn.Module):
    def __init__(self, num_features):
        super(FrequencyInstanceNorm2D, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.zeros(num_features, dtype=torch.complex64))

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=(-2, -1), keepdim=True)
        var = torch.var(inputs, dim=(-2, -1), unbiased=False, keepdim=True)
        freq_normalized = (inputs - mean) / torch.sqrt(var + 1e-5)
        freq_output = self.gamma.view(1, -1, 1, 1) * freq_normalized + self.beta.view(1, -1, 1, 1)
        return freq_output



def complex_to_real_imag(complex_tensor):
    real_part = complex_tensor.real
    imag_part = complex_tensor.imag
    return real_part,imag_part,torch.cat((real_part, imag_part), dim=1)

class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.pool_real = nn.AdaptiveAvgPool2d(output_size)
        self.pool_imag = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, inputs):
        real_part, imag_part, _ = complex_to_real_imag(inputs)
        pooled_real = self.pool_real(real_part)
        pooled_imag = self.pool_imag(imag_part)
        pooled_output = torch.complex(pooled_real, pooled_imag)
        return pooled_output  

class FrequencyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FrequencyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filters = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.complex64))
        self.Initialization()

    def Initialization(self):
        bound = 1 / (self.in_features ** 0.5)
        nn.init.uniform_(self.filters.real, -bound, bound)
        nn.init.uniform_(self.filters.imag, -bound, bound)
    def forward(self, inputs):
        freq_output = inputs.unsqueeze(1) * self.filters 
        freq_output = freq_output.sum(-1)
        return freq_output



class ComplexAdaptiveMaxPool2d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_real = nn.AdaptiveMaxPool2d(output_size)
        self.pool_imag = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        real = x.real
        imag = x.imag
        pooled_real = self.pool_real(real)
        pooled_imag = self.pool_imag(imag)
        pooled = torch.complex(pooled_real, pooled_imag)
        return pooled

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        real = self.dropout(x.real)
        imag = self.dropout(x.imag)
        return torch.complex(real, imag)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = FrequencyConv2D(in_channels, out_channels, kernel_size)
        self.bn1 = FrequencyInstanceNorm2D(out_channels)
        self.conv2 = FrequencyConv2D(out_channels, out_channels, kernel_size)
        self.bn2 = FrequencyInstanceNorm2D(out_channels)

        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                FrequencyConv2D(in_channels, out_channels, kernel_size=1),
                FrequencyInstanceNorm2D(out_channels)
            )
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = Log_Magnitude(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = Log_Magnitude(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = FrequencyConv2D(1, 64, kernel_size=1)
        self.bn1 = FrequencyInstanceNorm2D(64)
        
        self.layer1 = self._make_layer(64, 64, kernel_size=1, num_blocks=1)
        self.layer2 = self._make_layer(64, 128, kernel_size=1, num_blocks=1)
        self.layer3 = self._make_layer(128, 256, kernel_size=1, num_blocks=1)
        self.layer4 = self._make_layer(256, 512, kernel_size=1, num_blocks=1)
        
        self.pooling_layer = ComplexAdaptiveAvgPool2d(output_size=(16, 16))
        self.pooling_layer1 = ComplexAdaptiveAvgPool2d(output_size=(8, 8))
        self.pooling_layer2 = ComplexAdaptiveAvgPool2d(output_size=(4, 4))
        self.pooling_layer3 = ComplexAdaptiveAvgPool2d(output_size=(2, 2))
        
        
        self.avgpool = ComplexAdaptiveMaxPool2d(output_size=(1, 1))
        self.fc2 = FrequencyLinear(512, 5)
        self.dropout = ComplexDropout(p=0.4)
        
    
    def _make_layer(self, in_channels, out_channels, kernel_size, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        return nn.Sequential(*layers)
    

    
    def forward(self, x):
        '''magnitude = torch.abs(x)  
        phase = torch.angle(x)    
        magnitude_only = torch.polar(magnitude, torch.zeros_like(phase))
        phase_only = torch.polar(torch.ones_like(magnitude), phase)
        x=phase_only'''

        x = Log_Magnitude(self.bn1(self.conv1(x)))
        x = self.pooling_layer(x)
        

        x = self.layer1(x)
        x = self.pooling_layer1(x)
        

        x = self.layer2(x)
        x = self.pooling_layer2(x)
       

        x = self.layer3(x)
        x = self.pooling_layer3(x)
        

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc2(x)
               
        return x



mitbih_train1 = '/home/mainak/MDU/New FCNN/Ablation/data/ECG/mitbih_train.csv'  
mitbih_test1  = '/home/mainak/MDU/New FCNN/Ablation/data/ECG/mitbih_test.csv'  
mitbih_test = pd.read_csv(mitbih_test1, header=None)
mitbih_train = pd.read_csv(mitbih_train1, header=None)


mitbih_labels = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
mitbih_reverse_labels = {v:k for k,v in mitbih_labels.items()}
mitbih_train_labels = mitbih_train.iloc[:,-1].replace(mitbih_reverse_labels)


X_mitbih_train, X_mitbih_val, y_mitbih_train, y_mitbih_val = train_test_split(
    mitbih_train.iloc[:,:-1].values, mitbih_train.iloc[:,-1].values, test_size=0.1, random_state=42)
X_mitbih_test = mitbih_test.iloc[:,:-1].values
y_mitbih_test = mitbih_test.iloc[:,-1].values

X_mitbih_train = X_mitbih_train[:, np.newaxis, :, np.newaxis]
X_mitbih_val = X_mitbih_val[:, np.newaxis, :, np.newaxis]
X_mitbih_test = X_mitbih_test[:, np.newaxis, :, np.newaxis]

class FFTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long().squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data, label = self.X[idx], self.y[idx]
        complex_data = torch.fft.fft(torch.squeeze(data, -1), dim=-1)
        complex_data = torch.unsqueeze(complex_data, -1)
        return complex_data, label


trainset_fft = FFTDataset(X_mitbih_train, y_mitbih_train)
valset_fft = FFTDataset(X_mitbih_val, y_mitbih_val)
testset_fft = FFTDataset(X_mitbih_test, y_mitbih_test)


print("Training set size:", len(trainset_fft))
print("Validation set size:", len(valset_fft))
print("Test set size:", len(testset_fft))



batch_size = 32
train_loader = DataLoader(trainset_fft, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(valset_fft, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(testset_fft, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr) 

train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=700)


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()