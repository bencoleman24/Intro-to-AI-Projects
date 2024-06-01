import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training=True):
    transform_to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    compose_transforms = transforms.Compose([transform_to_tensor, normalize])
    
    if training:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=compose_transforms)
    else:
        dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=compose_transforms)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    
    return loader

def build_model():
    flatten_layer = nn.Flatten()
    first_linear_layer = nn.Linear(784, 128)
    first_relu = nn.ReLU()
    second_linear_layer = nn.Linear(128, 64)
    second_relu = nn.ReLU()
    third_linear_layer = nn.Linear(64, 10)
    
    model_layers = [flatten_layer, first_linear_layer, first_relu,
                    second_linear_layer, second_relu, third_linear_layer]
    model = nn.Sequential(*model_layers)
    
    return model

def train_model(model, train_loader, criterion, T):
    learning_rate = 0.001
    momentum = 0.9
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    for epoch in range(T):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / total
        epoch_acc = 100 * correct / total
        
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total}({epoch_acc:.2f}%) Loss: {epoch_loss:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            
            if show_loss:
                loss = criterion(outputs, labels)
                test_loss += loss.item() * data.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    if show_loss:
        average_loss = test_loss / total
        print(f'Average loss: {average_loss:.4f}')
    
    print(f'Accuracy: {accuracy:.2f}%')

def predict_label(model, test_images, index):
    model.eval()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    img = test_images[index].unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img)
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, 3)
        
        top_probs = top_probs.squeeze().tolist()
        top_idxs = top_idxs.squeeze().tolist()
        
        for i in range(3):
            print(f'{class_names[top_idxs[i]]}: {100 * top_probs[i]:.2f}%')


if __name__ == '__main__':
