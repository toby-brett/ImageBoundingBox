import cv2
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def process(batch_size, size=201):

    image_path = 'TUGraz_person'
    text_path = 'TUGraz_person_text'

    X_data = []
    y_data = []

    image_files = sorted(os.listdir(image_path))
    text_files = sorted(os.listdir(text_path))

    j = 0

    for i in range(batch_size):

        X_data.append([])
        y_data.append([])

        for image, text in zip(image_files[j:j+batch_size], text_files[j:j+batch_size]):

            txt_path = f'TUGraz_person_text/{text}'
            im_path = f'TUGraz_person/{image}'

            initial = cv2.imread(im_path)
            array = cv2.resize(initial, (size, size))

            scaleX = array.shape[1] / initial.shape[1]
            scaleY = array.shape[0] / initial.shape[0]

            with open(txt_path, 'r') as file:
                lines = file.readlines()

            pattern = r'Bounding box.*?: \((\d+), (\d+)\) - \((\d+), (\d+)\)'

            # r means raw string, so take every char as literal
            # Bounding box for object 1: looks for this exact phrase
            # . - matches any char except a new line
            # *? - matches the minimum it needs to make the pattern work
            # \( \) are parenthesis in text
            # \d matches a single digit
            # d+ means one or more digits
            # () parenthesis around the (\d+) means a caputre group which we can retreive later wth .groups
            # - hyphen that seperates the two coords in the text

            found_bbox = False
            for line in lines:

                if found_bbox:
                    break

                match = re.search(pattern, line)
                if match:
                    xmin, ymin, xmax, ymax = map(int, match.groups()) # map turns the groups into intagers

                    X_data[-1].append(array)
                    y_data[-1].append([xmin*scaleX, ymin*scaleY, xmax*scaleX, ymax*scaleY])

                    found_bbox = True

        j += 10

    X_tensor = torch.tensor(np.array(X_data)) / 255.0
    y_tensor = torch.tensor(np.array(y_data))
    return X_tensor, y_tensor

X_data, y_data = process(10)

class Model(nn.Module):
    def __init__(self, im_size):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 5, (3, 3), 1, 1) #  B, 5, 201, 201
        self.conv2 = nn.Conv2d(5, 5, (3, 3), 2, 0) #  B, 5, 100, 100
        self.conv3 = nn.Conv2d(5, 5, (3, 3), 1, 1) #  B, 5, 100, 100

        self.flattened_size =  5 * 100 * 100

        self.fcl1 = nn.Linear(self.flattened_size, 128)
        self.fcl2 = nn.Linear(128, 64)
        self.fcl3 = nn.Linear(64, 4) # x1, y1, x2, y2 - a regression task

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # flattens to B, 5*42*42
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)

        return x # logits for regression

model = Model(201).double()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(epochs):
    for epoch in range(epochs):
        for X, y in zip(X_data, y_data):

            X = X.permute(0, 3, 1, 2).double() # batch, channels, H, W
            preds = model(X)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train(300)

def test():
    for X, y in zip(X_data, y_data):

        X = X.permute(0, 3, 1, 2).double()  # batch, channels, H, W
        preds = model(X)

        for i in range(9):




            image = X[i].permute(1, 2, 0).numpy() #HWC

            box = preds[i].detach().numpy().astype(int)
            box2 = y[i].detach().numpy().astype(int)

            cv2.line(image, (box[0], box[1]), (box[2], box[1]), (255, 0, 0), 2)
            cv2.line(image, (box[2], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.line(image, (box[2], box[3]), (box[0], box[3]), (255, 0, 0), 2)
            cv2.line(image, (box[0], box[3]), (box[0], box[1]), (255, 0, 0), 2)

            cv2.line(image, (box2[0], box2[1]), (box2[2], box2[1]), (0, 255, 0), 2)
            cv2.line(image, (box2[2], box2[1]), (box2[2], box2[3]), (0, 255, 0), 2)
            cv2.line(image, (box2[2], box2[3]), (box2[0], box2[3]), (0, 255, 0), 2)
            cv2.line(image, (box2[0], box2[3]), (box2[0], box2[1]), (0, 255, 0), 2)

            cv2.imshow("name", image)
            cv2.waitKey(0)

test()
