import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def train(model, criterion, optimizer, train_data, test_data, epochs=1, noise=True):
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    valloader = DataLoader(test_data, batch_size=1, shuffle=True) #batch=512
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    best_val = 100
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            #inputs = inputs[:, None]
            #labels = labels[:, None]
            #model.train()  # put model to training mode
            #inputs = inputs.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #labels = labels.to(device=device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, out_shift = model(inputs, noise=noise)
            loss = criterion(outputs, labels)

            if epoch>=0:
                l2_lambda = 0.0
            else:
                l2_lambda=1.0
            l2_norm = criterion(out_shift, inputs)

            loss = loss + l2_lambda * l2_norm


            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /10 :.5f}')
                running_loss = 0.0
                check_accuracy(valloader, model, 'train', device, noise=noise)
                print()

        distortion_val = check_accuracy(valloader, model, 'test', device, noise=noise)
        if distortion_val < best_val:
            best_val = distortion_val
            best_model = model
        scheduler.step()


    print('Finished Training')
    return best_model, best_val


def check_accuracy(loader, model, mode, device, noise=True):
    print('Checking accuracy')
    distortion = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        loss = nn.MSELoss(reduction='sum')
        val_num = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            #x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #y = y.to(device=device, dtype=torch.long)
            #x = x[:, None]
            #y = y[:, None]
            val_num += x.size(0)
            if mode == 'train':
                preds, _ = model(x, training=True, noise=noise)
            else:
                preds, _ = model(x, training=False, noise=noise)
            distortion += loss(y, preds)
        distortion /= val_num
        print('Got distortion of %f' % (distortion))
    if mode == 'test':
        return distortion


def shift_test(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_vec = torch.linspace(-6.4, 6.4, 1000, device=device)  # input vec
    input_vec = input_vec[:, None]
    model = model.to(device=device)
    _, output_vec = model(input_vec)
    #plt.plot(input_vec.detach().cpu().numpy(), output_vec.detach().cpu().numpy())
    #plt.show()
    return torch.squeeze(input_vec), torch.squeeze(output_vec)
