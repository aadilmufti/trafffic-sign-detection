#using the IDSIA MCDNN model architecture with one extra batch normalization layer of 350 maps

#model = Net().to(device)

class Net(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return x
        
"""
#From our notebook titled: Traffic Sign Detection with MCDNN with data augmentation
#In the notebook the model class is named TrafficSignNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
n_epochs = 30

model = Net().to(device)
optimizer = optim.Adamax(model.parameters(), lr=0.0005)
data_dict_1 = trainer(n_epochs, model, criterion, optimizer, train_loader, valid_loader)

n_epochs = 100
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
data_dict_2 = trainer(n_epochs, model, criterion, optimizer, train_loader, valid_loader)

"""
        
