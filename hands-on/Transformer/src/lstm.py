import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from sklearn.preprocessing import MinMaxScaler
#vizualize the datasets in seaborn
sns.get_dataset_names()

# #loading the dataset
# flight_dataset = sns.load_dataset('flights')
# #To vizualize the loaded dataset
# flight_dataset

# #plotting the data
# plt.plot(flight_dataset['passengers'])
# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.xlabel('Months')
# plt.grid(True)
# plt.show()

df = pd.read_csv("data_pred_3.csv",sep=",")
df.columns = ["date", "youbi", "actions"] # ["datetime","id","value"]
from datetime import datetime as dt
df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y%m%d%H%M%S"))

# plt.scatter(df['youbi'], df['actions'])
plt.plot(df['date'], df['actions'])
plt.title('youbi vs actions')
plt.ylabel('action')
plt.xlabel('date')
plt.grid(True)
plt.show()

#change the values of passengers to float values for prcoessing further
# passenger_values = flight_dataset['passengers'].values.astype(float)
values = df['actions'].values.astype(float)

#train, test splitting the dataset train = (0,132) test = (132,144)
split_size = 12
# train_values = passenger_values[:-split_size]
# test_values = passenger_values[-split_size:]
train_values = values[:-split_size]
test_values = values[-split_size:]
#to evaluate oor normalize the values between -1 and 1
mm_scaler = MinMaxScaler(feature_range=(-1, 1)) 
mm_scaler = mm_scaler.fit(np.expand_dims(train_values, axis=1))
train_data = mm_scaler.transform(np.expand_dims(train_values, axis=1))

print(train_data.shape, train_data[0:5])

num = 21


#These created train and test data should be split the sequence into input (X) and output (y) values depending on sequence length
def split_sequences(sequences_data,sequence_length):
    X,y = list(),list()
    for i in range(len(sequences_data)-sequence_length-1):
        # find the end of this pattern
        end_ix = i + sequence_length
        #gather input and output parts of the pattern
        seq_x, seq_y = sequences_data[i:(end_ix)],sequences_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
#Obtaining X and y values of train data
X_train, y_train = split_sequences(sequences_data=train_data,sequence_length = num) # 3) # 12)
#Viewing first element of input train value

print(X_train[0], X_train.shape)
# print(X_train, X_train.shape)

#reshaping to required
X_train = X_train.reshape(len(X_train), num) # 3) # 12)
print(X_train[0], X_train.shape)

#Viewing first element of output train value
print(y_train[0], y_train.shape)

#Converting train data from numpy array into Pytorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
print(X_train.shape, y_train.shape)

# num = 41


#Defining a class of future data prediction using 'LSTM' as of self.lstm
class FutureDataPrediction(nn.Module):
    def __init__(self, feature_size, hidden_layer):
        super(FutureDataPrediction, self).__init__()
        self.hidden_layer = hidden_layer
        
        self.lstm = nn.LSTM(input_size=feature_size,hidden_size=hidden_layer)
        self.linear = nn.Linear(in_features=hidden_layer, out_features=1)
        self.hidden = (
            torch.zeros(1,1, self.hidden_layer),
            torch.zeros(1,1,self.hidden_layer))
    
    def forward(self, data):
        lstm, self.hidden = self.lstm(data.view(len(data), 1, -1),self.hidden)
        previous_time_step = lstm.view(len(data), -1)
        out = self.linear(previous_time_step)
        return out[-1]
   
class FutureDataPrediction(nn.Module):
    def __init__(self, feature_size, hidden_layer, seq_len):
        super(FutureDataPrediction, self).__init__()
        self.hidden_layer = hidden_layer
        self.seq_len = seq_len
        
        
        self.lstm = nn.LSTM(input_size=feature_size,hidden_size=hidden_layer)
        self.linear = nn.Linear(in_features=hidden_layer, out_features=1)
    
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(1, self.seq_len, self.hidden_layer),
            torch.zeros(1, self.seq_len, self.hidden_layer))
    
    def forward(self, data):
        lstm, self.hidden = self.lstm(data.view(len(data), self.seq_len, -1),self.hidden)
        previous_time_step = lstm.view(self.seq_len, len(data), self.hidden_layer)[-1]
        out = self.linear(previous_time_step)
        return out
   
#initialise modelparameters
model = FutureDataPrediction(
    feature_size=1,
    hidden_layer=100,
    seq_len=num) # 3) # 12)
criterion = torch.nn.MSELoss() #loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #optimizer
print(model)

#The designed model structure can be vizualized
torch.save(model.state_dict(), 'LSTM_Model.pt') #saving the model
model.load_state_dict(torch.load('LSTM_Model.pt')) #loading the model

#defining train section of the model to attain the best predictions
def training(model,X_train,y_train,n_epochs):
    model.train()
    train_losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        model.reset_hidden_state() #the state has to be initialise to the start position after an epoch start.
        y_pred = model(X_train)
        loss = criterion(y_pred.float(), y_train) #attain loss to update weights.
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
        # if epoch % 1 == 0:
            print(f'Epoch {epoch} train loss: {loss.item()}')
            
        loss.backward()
        optimizer.step()
    return train_losses


train_losses = training(model,X_train,y_train,n_epochs=500) # 400)

# Plotting loss values
'loss values during training for each epoch calculation'
plt.figure(figsize=(8,6))
plt.plot(train_losses, label='Training')    
plt.title("loss vs epoch",fontsize=20)
plt.xlabel("epoch",fontsize=15)
plt.ylabel("loss",fontsize=15)
plt.legend(fontsize=15)
plt.show()

# num = 21
#Predicting data of future months
future_months = num # 3 # 12 #as the test data remainded is 12
sequence_length = num # 3 # 12
test_data = torch.FloatTensor(train_data[-sequence_length:].tolist()).reshape(1,num,1) # 12,1)
print(test_data.shape, test_data) #defines next 12 months data i.e. indices: [133,134....144]

#validating values for the test data for future prediction
def testing(model,test_data,predict_days,sequence_length):
    model.eval()
    with torch.no_grad():
        test_sequence = test_data #to send the last data of X as input to network to predict future
        predicts = []

        for _ in range(predict_days):
            y_pred = model(test_sequence)
            predict = torch.flatten(y_pred).item() #flatten the values 
            predicts.append(predict)
            
            test_seq = test_sequence.numpy().flatten() #flatten old sequence to connect with new sequence
            test_seq = np.append(test_seq, [predict]) #obtain new predicted as sequence
            test_seq = test_seq[1:] #to perform last values
            testing_seq = torch.as_tensor(test_seq).view(1, sequence_length, 1).float()
    return predicts
predict_months = num # 3 # 12 #next months data cases can be an estimated confirm
predict_passengers = testing(model,test_data,predict_months,sequence_length = num) # 3) # 12) #same sequence_length from before data split
print(predict_passengers)

#reverse values from 0 to 1 by minmaxscaler to original values
future_passengers = mm_scaler.inverse_transform(np.expand_dims(predict_passengers, axis=0)).flatten() 
index = np.arange(num, num+21, 1) # 0, num, 1) # 132, 144, 1)
print(index,future_passengers)

plt.title('Date vs Action')
plt.ylabel('Action')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
# plt.plot(df['date'], df['actions'])
plt.plot(df['actions'])
plt.plot(index, future_passengers)
plt.legend(['True Data','Predicted Data'])
plt.show()