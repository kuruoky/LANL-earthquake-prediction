import os
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def init_spark():
  spark = SparkSession.builder.appName("model").config('spark.driver.memory', '4g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()

df_f = spark.read.parquet(os.path.join("data", "train.win.vector.fbin.2.parquet"))
df_y = spark.read.parquet(os.path.join("data", "train.win.target.parquet"))

df_f = df_f.selectExpr("*").drop("_c0")
df_y = df_y.selectExpr("seg AS seg2", "*").drop("seg")

df_train = df_f
df_train = df_train.join(df_y, df_train.seg.cast(IntegerType()) == df_y.seg2.cast(IntegerType())).drop("seg2")

df_train.printSchema()



# test and training set
trainingData = df_train.selectExpr("*").where("seg < 3145")
testData = df_train.selectExpr("*").where("seg >= 3145")

x_train = torch.Tensor([])
x_test = torch.Tensor([])
y_train = torch.Tensor([])
y_test = torch.Tensor([])
for j in range(8):
    x_temp = trainingData.select("f_sl" + str(j)).orderBy("seg").collect()
    x_temp = torch.Tensor(np.array(x_temp).astype('float32')).reshape(-1, 1, 26)
    if x_train.shape[0] == 0:
        x_train = x_temp
    else:
        x_train = torch.cat([x_train, x_temp], 1)

    x_temp = testData.select("f_sl" + str(j)).orderBy("seg").collect()
    x_temp = torch.Tensor(np.array(x_temp).astype('float32')).reshape(-1, 1, 26)
    if x_test.shape[0] == 0:
        x_test = x_temp
    else:
        x_test = torch.cat([x_test, x_temp], 1)

    y_temp = trainingData.select("y" + str(j)).orderBy("seg").collect()
    y_temp = torch.Tensor(np.array(y_temp).astype('float32')).reshape(-1, 1)
    if y_train.shape[0] == 0:
        y_train = y_temp
    else:
        y_train = torch.cat([y_train, y_temp], 1)

    y_temp = testData.select("y" + str(j)).orderBy("seg").collect()
    y_temp = torch.Tensor(np.array(y_temp).astype('float32')).reshape(-1, 1)
    if y_test.shape[0] == 0:
        y_test = y_temp
    else:
        y_test = torch.cat([y_test, y_temp], 1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

Y_max = 16.103195566
y_train = y_train/Y_max
y_test = y_test/Y_max

class RecN(nn.Module):
    def __init__(self):
        super(RecN, self).__init__()
        self.num_layers = 2
        self.hidden_dim = 12
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.20)
        self.rnn = nn.GRU(26, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.15)
        self.linear1 = nn.Linear(self.hidden_dim, 96)
        self.linear2 = nn.Linear(96, 56)
        self.out = nn.Linear(56, 1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

    def forward(self, x, h, debug=False):
        if debug : print("x-in       \t", x.shape)
        seq_size = x.shape[1]
        x_dim = x.shape[2]
        j, h = self.rnn(x, h)
        if debug : print("hidden out \t", h.shape)
        if debug : print("lstm out   \t", j.shape)
        j = self.dropout(j)
        j = j[:,-1,:]
        if debug : print("format    \t", j.shape)
        j = self.linear1(j)
        if debug : print("l1        \t", j.shape)
        j = self.relu(j)
        j = self.dropout(j)
        j = self.linear2(j)
        if debug : print("l2        \t", j.shape)
        j = self.relu(j)
        j = self.dropout(j)
        y = self.out(j)
        if debug : print("l-out     \t", y.shape)
        return y, h

model = RecN()

x = x_train[1].unsqueeze(0)
print(x.shape)
h = model.init_hidden(x.shape[0])
print(h.shape)
y_hat,_ = model(x, h, debug=True)
print(y_hat.shape)
print(y_hat)

model = RecN()
loss_f = torch.nn.L1Loss(reduction="mean")
optimiser = torch.optim.Adam(model.parameters(), lr=0.001,
                             betas=(0.92, 0.999), eps=1e-10, amsgrad=True,
                             weight_decay=0.0000)

num_epochs = 600
loss_train = 0
valid_loss = 100000
best_model = RecN()
hist_t, hist_v = [], []
for t in range(num_epochs):

    model.train()
    hidden = model.init_hidden(x_train.shape[0])
    for i in range(x_train.shape[1]):
        optimiser.zero_grad()
        y_hat, hidden = model(x_train[:, i].view(-1, 1, 26), hidden)
        loss = loss_f(y_hat, y_train[:, i].view(-1, 1))
        loss.backward()
        optimiser.step()
        hidden = hidden.detach()
    if t % 50 == 0: loss_train = (loss.item() * Y_max)
    hist_t.append(loss.item() * Y_max)

    model.eval()
    hidden = model.init_hidden(x_test.shape[0])
    for i in range(x_test.shape[1]):
        with torch.no_grad():
            y_hat, hidden = model(x_test[:, i].view(-1, 1, 26), hidden)
    loss = loss_f(y_hat, y_test[:, i].view(-1, 1))
    if loss.item() < valid_loss:
        valid_loss = loss.item()
        best_model.load_state_dict(model.state_dict())

    if t % 50 == 0: print(
        "#{:04} \tTrn.: {:.06f}\t Vld.: {:.06f}\t Bst.: {:.06f}".format(t, loss_train, loss.item() * Y_max,
                                                                        valid_loss * Y_max))
    hist_v.append(loss.item() * Y_max)

torch.save(best_model.state_dict(), "model.3.0.pytorch.rnn.pt")
print("done!")

best_model.eval()
hidden = best_model.init_hidden(x_test.shape[0])
for i in range(x_test.shape[1]):
    with torch.no_grad():
        y_hat, hidden = best_model(x_test[:,i].view(-1,1,26), hidden)

fig, ax = plt.subplots(figsize=(20,8))
plt.plot(y_hat.squeeze(0).numpy()*Y_max, alpha=0.8, label="predictions")
plt.plot(y_test[:,i].view(-1,1).squeeze(0).numpy()*Y_max, label="labels")
title = ax.set_title("Prediction vs. ground truth", loc="left")
legend = plt.legend(loc="upper left")
plt.savefig("rnn.png")

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test[:,i].view(-1,1).squeeze(0).numpy()*Y_max, y_hat.squeeze(0).numpy()*Y_max)

mean_absolute_error(y_test[:,i].view(-1,1).squeeze(0).numpy()*Y_max, y_hat.squeeze(0).numpy()*Y_max)