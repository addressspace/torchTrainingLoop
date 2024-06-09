import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torchFit import trainingLoop
device = 'cuda'

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28*28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
        
 
 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
validation_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




model = SimpleNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(optimizer)


'''
def trainingLoop(train_loader, num_epochs, criterion, optimizer, validation_loader = None, lr_decay_factor = 0.9, modelSave = True, homeDir = False):
	newpaths = 'modelDicts', 'lossArray', 'myTrashed'
	for newpath in newpaths:
		if homeDir:
			newpath = homeDir + '/' + newpath
		if not os.path.exists(newpath):
			os.makedirs(newpath)
	
	initial_lr = optimizer.state_dict()['param_groups'][0]['lr']
	trainingLossArray = []
	validationLossArray = []

	for epoch in range(num_epochs):
		model.train()
		trainingRunningLoss = 0.0
		for xtrain, ytrain in train_loader:
			# Zero the parameter gradients
			optimizer.zero_grad()
	        # Forward pass
			trainPred = model(xtrain)
			trainingLoss = criterion(trainPred, ytrain)
			# Backward pass and optimize
			trainingLoss.backward()
			optimizer.step()
			trainingRunningLoss += trainingLoss.item()
	
		#if modelSave:
				#torch.save()
		
		currentEpochTrainingLoss = trainingRunningLoss/len(train_loader)
		trainingLossArray.append(currentEpochTrainingLoss)
		trainingTextOutput = f'|Training Loss -- {currentEpochTrainingLoss:.4f}|'
		

		lrTextOutput = ''
		if lr_decay_factor:
			CurrentLR = optimizer.state_dict()['param_groups'][0]['lr']
			lrTextOutput = f'|Current Learning Rate -- {CurrentLR}|'
			#learning rate updation
			new_lr = initial_lr / (lr_decay_factor * (epoch + 1))
			for param_group in optimizer.param_groups:
				param_group['lr'] = new_lr
				#initial_lr = new_lr
			

	    
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {trainingRunningLoss/len(train_loader):.4f}")
		print(currentEpochTrainingLoss)


		validationTextOutput = ''
		if validation_loader != None:
			# Validation step
			model.eval()
			validationRunningLoss = 0.0
			with torch.no_grad():
				for xvalid, yvalid in validation_loader:
					validPred = model(xvalid)
					validationLoss = criterion(validPred, yvalid)
					validationRunningLoss += validationLoss.item()
			currentEpochValidationLoss = validationRunningLoss/len(validation_loader)
			validationLossArray.append(currentEpochValidationLoss)
			validationTextOutput = f'|Validation Loss -- {currentEpochValidationLoss:.4f}|'

		
		print(currentEpochValidationLoss)
'''
trainingLoop(model, trainingDataLoader, 216, loss_fn, optimizer, resume=True, lr_decay_factor = 0.9, validation_loader=validationDataLoader, device=device)