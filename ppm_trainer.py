import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model, criterion, optimizer, train_data, test_data, epochs=1):
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = DataLoader(test_data, batch_size=64, shuffle=True) #doto: cancel batch

    #model = model.to(device=device)  # move the model parameters to CPU/GPU
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #model.train()  # put model to training mode
            #inputs = inputs.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #labels = labels.to(device=device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /10 :.3f}')
                running_loss = 0.0
                check_accuracy(valloader, model)
                print()

    print('Finished Training')


def check_accuracy(loader, model):
    print('Checking accuracy')
    distortion = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            #x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #y = y.to(device=device, dtype=torch.long)
            preds = model(x)
            MSELoss = nn.MSELoss()
            distortion += MSELoss(y, preds)
        print('Got distortion of %f' % (distortion))
