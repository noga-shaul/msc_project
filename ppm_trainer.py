import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def train(model, criterion, optimizer, train_data, test_data, epochs=1):
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = DataLoader(test_data, batch_size=64, shuffle=True) #doto: cancel batch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    #model = model.to(device=device)  # move the model parameters to CPU/GPU
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #inputs = inputs[:, None]
            #labels = labels[:, None]
            #model.train()  # put model to training mode
            #inputs = inputs.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #labels = labels.to(device=device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /10 :.3f}')
                running_loss = 0.0
                check_accuracy(valloader, model, 'train')
                print()
        scheduler.step()

    print('Finished Training')


def check_accuracy(loader, model, mode):
    print('Checking accuracy')
    distortion = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        loss = nn.MSELoss(reduction='sum')
        val_num = 0
        for x, y in loader:
            #x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #y = y.to(device=device, dtype=torch.long)
            #x = x[:, None]
            #y = y[:, None]
            val_num += x.size(0)
            preds, _ = model(x)
            distortion += loss(y, preds)
        distortion /= val_num
        print('Got distortion of %f' % (distortion))
    if mode == 'test':
        return distortion


def shift_test(model):
    input_vec = torch.linspace(-6.4, 6.4, 1000)  # input vec
    input_vec = input_vec[:, None]
    _, output_vec = model(input_vec)
    plt.plot(input_vec.detach().numpy(), output_vec.detach().numpy())
    plt.show()