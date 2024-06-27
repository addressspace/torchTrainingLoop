import torch
import numpy as np
import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
import time







def save_checkpoint(model, optimizer, epoch, file_path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)










def load_checkpoint(model, optimizer, file_path='checkpoint.pth'):
    try:
        try:
            checkpoint = torch.load(file_path)
        except:
            checkpoint = torch.load(file_path[:-4] + '-backup.pth')
			
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    except FileNotFoundError:
        print("No checkpoint found, starting training from scratch")
        return 0
    except KeyError as e:
        print(f"Key error: {e}. Starting training from scratch")
        return 0











def trainingLoop(model, xtrainingData, ytrainingData, trainingGrad, xvalidationData, yvalidationData, validationGrad, num_epochs, criterion, optimizer, validation_loader = True, lr_decay_factor = False, modelSave = True, homeDir = False, resume=False, device='cpu', visualization=True):
	full_xtrain = xtrainingData.squeeze().unsqueeze(-1).permute(0, 3, 1, 2).to(device)
	full_gtrain = trainingGrad.squeeze().unsqueeze(-1).permute(0, 3, 1, 2).to(device)
	full_ytrain = ytrainingData.squeeze().unsqueeze(-1).permute(0, 3, 1, 2).to(device)
	full_xvalid = xvalidationData.squeeze().unsqueeze(-1).permute(0, 3, 1, 2).to(device)
	full_gvalid = validationGrad.squeeze().unsqueeze(-1).permute(0, 3, 1, 2).to(device)
	full_yvalid = yvalidationData.squeeze().unsqueeze(-1).permute(0, 3, 1, 2).to(device)
	print(full_xtrain.shape)
	BATCH_SIZE = 16
	trainingData = torch.utils.data.TensorDataset(full_xtrain, full_ytrain, full_gtrain)
	trainingDataLoader = torch.utils.data.DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True)
	validationData = torch.utils.data.TensorDataset(full_xvalid, full_yvalid, full_gvalid)
	validationDataLoader = torch.utils.data.DataLoader(validationData, batch_size=BATCH_SIZE, shuffle=True)
	train_loader = trainingDataLoader
	validation_loader = validationDataLoader



	if 	visualization:
		# Initialize figure
		fig = make_subplots(rows=1, cols=1)
		fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers+text', name='Training Loss', text=[], textposition='top right'))
		fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers+text', name='Validation loss', text=[], textposition='top right'))

		# Display the initial empty plot
		display(fig)

		# Initialize data arrays
		x_data = []



	newpaths = ['model_dicts', 'loss_array', 'myTrashed']
	for newpath in newpaths:
		if homeDir:
			newpath = homeDir + '/' + newpath
		if not os.path.exists(newpath):
			os.makedirs(newpath)
		
	modelLoadDir = f'{newpaths[2]}/lastModel.pth' #for resume training
	if homeDir:
		modelLoadDir = homeDir + '/' + modelLoadDir
	
	#load_checkpoint(model, optimizer, path=modelLoadDir)
	start_epoch = 0
	
	if resume:
		try:
			start_epoch = load_checkpoint(model, optimizer, modelLoadDir)
			print(f"Resuming training from epoch {start_epoch}")
		except FileNotFoundError:
			print("No checkpoint found, starting training from scratch")
		
		
		
	
	initial_lr = optimizer.state_dict()['param_groups'][0]['lr']
	trainingLossArray = []
	validationLossArray = []

	for epoch in range(start_epoch, num_epochs):
		model.train()
		trainingRunningLoss = 0.0
		for xtrain, ytrain, gtrain in train_loader:
			# Zero the parameter gradients
			optimizer.zero_grad()
	        # Forward pass
			trainPred = model(xtrain.to(device), gtrain.to(device))
			trainingLoss = criterion(trainPred, ytrain.to(device))
			# Backward pass and optimize
			trainingLoss.backward()
			optimizer.step()
			trainingRunningLoss += trainingLoss.item()
		
		modelDir = f'{newpaths[0]}/epoch-{epoch+1}.pth' #for saving all model
		lastModelDir = f'{newpaths[2]}/lastModel.pth' #for resume training
		trainingLossDir =  f'{newpaths[1]}/trainingLoss'
		validationLossDir =  f'{newpaths[1]}/validationLoss'
		if homeDir:
			modelDir = homeDir + '/' + modelDir
			lastModelDir = homeDir + '/' + lastModelDir
			trainingLossDir = homeDir + '/' + trainingLossDir
			validationLossDir = homeDir + '/' + validationLossDir
		if modelSave:
			torch.save(obj=model.state_dict(), f=modelDir)
		if resume:
			if epoch%2 == 0:
				save_checkpoint(model, optimizer, epoch, lastModelDir)
			else:
				save_checkpoint(model, optimizer, epoch, lastModelDir[:-4] + '-backup.pth')
			
		currentEpochTrainingLoss = trainingRunningLoss/len(train_loader)
		trainingLossArray.append(currentEpochTrainingLoss)
		trainingTextOutput = f'| Training Loss -- {currentEpochTrainingLoss:.4f} |'
		np.save(trainingLossDir, trainingLossArray)
		

		lrTextOutput = ''
		if lr_decay_factor:
			CurrentLR = optimizer.state_dict()['param_groups'][0]['lr']
			lrTextOutput = f'| Current Learning Rate -- {CurrentLR} |'
			#learning rate updation
			new_lr = initial_lr / (lr_decay_factor * (epoch + 1))
			for param_group in optimizer.param_groups:
				param_group['lr'] = new_lr
				#initial_lr = new_lr
			

	    
		#print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {trainingRunningLoss/len(train_loader):.4f}")
		#print(currentEpochTrainingLoss)


		validationTextOutput = ''
		if validation_loader != None:
			# Validation step
			model.eval()
			validationRunningLoss = 0.0
			with torch.no_grad():
				for xvalid, yvalid, gvalid in validation_loader:
					validPred = model(xvalid.to(device), gvalid.to(device))
					validationLoss = criterion(validPred, yvalid.to(device))
					validationRunningLoss += validationLoss.item()
			currentEpochValidationLoss = validationRunningLoss/len(validation_loader)
			validationLossArray.append(currentEpochValidationLoss)
			np.save(validationLossDir, validationLossArray)
			validationTextOutput = f'| Validation Loss -- {currentEpochValidationLoss:.4f} |'

		
		if 	visualization:
			# Automatically assign integer x value
			x_data.append(epoch)

			# Simulate data update
			#y_data_sin.append(np.sin(i * np.pi / 10))  # Example data: sine wave
			#y_data_cos.append(np.cos(i * np.pi / 10))  # Example data: cosine wave

			# Prepare the text annotations for the last points
			text_1 = [''] * (len(trainingLossArray) - 1) + [f'{trainingLossArray[-1]:.5f}']
			text_2 = [''] * (len(validationLossArray) - 1) + [f'{validationLossArray[-1]:.5f}']

			# Update the plot
			with fig.batch_update():
				fig.data[0].x = x_data
				fig.data[0].y = trainingLossArray
				fig.data[0].text = text_1

				fig.data[1].x = x_data
				fig.data[1].y = validationLossArray
				fig.data[1].text = text_2
		
			# Clear the previous output and display the updated figure
			clear_output(wait=True)
			display(fig)


		print(f"Epoch [{epoch+1:03}/{num_epochs}] |" + trainingTextOutput + validationTextOutput + lrTextOutput)
	
	model.eval()
	with torch.inference_mode():
		mainTrainingOutput = model(full_xtrain, full_gtrain)
		mainValidationOutput = model(full_xvalid, full_gvalid)
	return mainTrainingOutput, mainValidationOutput
		
'''
#Usage Example
#trainingLoop(model, trainingDataLoader, 216, loss_fn, optimizer, resume=True, lr_decay_factor = 0.9, validation_loader=validationDataLoader, device=device)
'''