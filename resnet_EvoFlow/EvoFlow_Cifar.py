import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time
import yaml
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Custom CIFAR-10 Dataset class to load data from local files
class CustomCIFAR10(Dataset):
    def __init__(self, data_files, root_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for file_name in data_files:
            file_path = os.path.join(root_dir, file_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data.append(batch[b'data'])
                self.labels.extend(batch[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define paths to the CIFAR-10 files
root_dir = "/home/khan/Desktop/paper-6th/resnet"
train_files = [f"data_batch_{i}" for i in range(1, 6)]
test_files = ["test_batch"]

# Define transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
train_dataset = CustomCIFAR10(data_files=train_files, root_dir=root_dir, transform=transform_train)
test_dataset = CustomCIFAR10(data_files=test_files, root_dir=root_dir, transform=transform_test)

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Define EvoFlow optimizer
class EvoFlow(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.87, beta2=0.999, alpha=0.92, lambda_=0.002, 
                 eps=1e-7, evolve_freq=75, loss_fn=None):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, alpha=alpha, lambda_=lambda_, eps=eps)
        super(EvoFlow, self).__init__(params, defaults)
        self.evolve_freq = evolve_freq
        self.loss_fn = loss_fn
        self.t = 0

    def step(self):
        self.t += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['f'] = torch.zeros_like(p.data)
                    state['g_prev'] = torch.zeros_like(p.data)

                state['step'] += 1
                m, v, f, g_prev = state['m'], state['v'], state['f'], state['g_prev']
                beta1, beta2, alpha, epsilon = group['beta1'], group['beta2'], group['alpha'], group['eps']
                c_base = 0.7
                c_scale = 0.5

                grad_consistency = torch.sum(grad * g_prev) / (torch.norm(grad) * torch.norm(g_prev) + epsilon)
                beta1_t = beta1 * max(0.65, grad_consistency.item())
                alpha_t = alpha * min(1.0, grad_consistency.item() + 0.65)

                m.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                f.mul_(alpha_t).add_(grad, alpha=1 - alpha_t)
                v_hat = v / (1 - beta2 ** state['step'])
                c_t = c_base + c_scale * torch.tanh(torch.abs(grad - g_prev) / (torch.sqrt(v_hat) + epsilon))

                if self.t % self.evolve_freq == 0:
                    p_data_old = p.data.clone()
                    grad_norm_old = torch.norm(grad)
                    delta = torch.normal(0, 0.008, p.data.shape, device=p.data.device)
                    p.data.add_(delta)
                    grad_new = p.grad.data if p.grad is not None else grad
                    grad_norm_new = torch.norm(grad_new)
                    if grad_norm_new >= grad_norm_old:
                        p.data.copy_(p_data_old)

                update = c_t * (m / (torch.sqrt(v_hat) + epsilon) + group['lambda_'] * p.data)
                p.data.add_(-group['lr'] * update)
                state['g_prev'].copy_(grad)

# Training and evaluation function
def train_and_evaluate(optimizer_name, params=None):
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer based on name
    if optimizer_name == "EvoFlow":
        optimizer = EvoFlow(model.parameters(), lr=params["lr"], beta1=params["beta1"], beta2=params["beta2"], 
                            alpha=params["alpha"], lambda_=params["lambda_"], evolve_freq=params["evolve_freq"])
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    elif optimizer_name == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    elif optimizer_name == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, momentum=0.9)

    epochs = 100
    training_losses = []
    print(f"\nStarting training with {optimizer_name}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        training_losses.append(avg_loss)
        print(f"{optimizer_name}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0
    print(f"Evaluating {optimizer_name}")
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.topk(5, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
            correct_1 += correct[:1].reshape(-1).float().sum(0)
            correct_5 += correct[:5].reshape(-1).float().sum(0)
            total += targets.size(0)

    top1_accuracy = correct_1 / total
    top5_accuracy = correct_5 / total

    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    total_time = 0
    num_images = 1000
    print(f"Measuring inference time for {optimizer_name}")
    with torch.no_grad():
        for _ in range(num_images):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            total_time += (time.time() - start_time)
    inference_time_ms = (total_time / num_images) * 1000

    print(f"\nResults for {optimizer_name}:")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Inference Time: {inference_time_ms:.4f} ms")

    return top1_accuracy, top5_accuracy, inference_time_ms, training_losses

# Define optimizers and parameters
optimizers = {
    "EvoFlow": {"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "alpha": 0.85, "lambda_": 0.0001, "evolve_freq": 25},
    "AdamW": None,  # Using defaults: lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
    "SGD": None,    # Using defaults: lr=0.01, momentum=0.9, weight_decay=0.0001
    "RAdam": None,  # Using defaults: lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
    "RMSProp": None # Using defaults: lr=0.001, alpha=0.99, momentum=0.9
}

# Run experiments
results = {}
training_losses_dict = {}
print("Starting optimizer comparison experiment...")
for optimizer_name, params in optimizers.items():
    top1, top5, inference_time, training_losses = train_and_evaluate(optimizer_name, params)
    results[optimizer_name] = {
        "Top-1 Accuracy": top1.item(),
        "Top-5 Accuracy": top5.item(),
        "Inference Time (ms)": inference_time
    }
    training_losses_dict[optimizer_name] = training_losses

# Print comparison table
print("\nComparison Table:")
print("| Optimizer | Top-1 Accuracy | Top-5 Accuracy | Inference Time (ms) |")
print("|-----------|----------------|----------------|---------------------|")
for optimizer_name, metrics in results.items():
    print(f"| {optimizer_name:<9} | {metrics['Top-1 Accuracy']:.4f}         | {metrics['Top-5 Accuracy']:.4f}         | {metrics['Inference Time (ms)']:.4f}            |")

# Save results
with open('optimizer_comparison.yaml', 'w') as f:
    yaml.dump(results, f)
print("\nResults saved to 'optimizer_comparison.yaml'")

# Plot training losses
def plot_training_loss(training_losses_dict):
    print("Generating Training Loss Over Epochs plot...")
    plt.figure(figsize=(10, 6))
    for optimizer_name, losses in training_losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=optimizer_name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_comparison.png')
    plt.close()
    print("Saved 'training_loss_comparison.png'")

plot_training_loss(training_losses_dict)
print("\nExperiment completed! Check 'optimizer_comparison.yaml' and 'training_loss_comparison.png' for results.")