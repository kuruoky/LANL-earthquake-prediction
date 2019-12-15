import torch
import os
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
import numpy as np
import torch.nn as nn
import pandas as pd

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
state_dict = torch.load("model.3.0.pytorch.rnn.pt")
model.load_state_dict(state_dict)

def init_spark():
  spark = SparkSession.builder.appName("output").config('spark.driver.memory', '4g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()

df_f = spark.read.parquet(os.path.join("data", "test.win.vector.fbin.2.parquet"))

df_f = df_f.selectExpr("*").drop("_c0")
df_f.seg.cast(IntegerType())



x_test = torch.Tensor([])

for j in range(8):
    x_temp = df_f.select("f_sl" + str(j)).orderBy("seg").collect()
    x_temp = torch.Tensor(np.array(x_temp).astype('float32')).reshape(-1, 1, 26)
    if x_test.shape[0] == 0:
        x_test = x_temp
    else:
        x_test = torch.cat([x_test, x_temp], 1)

print(x_test.shape)

max = np.max(np.asarray(x_test))

model.eval()
hidden = model.init_hidden(x_test.shape[0])
for i in range(x_test.shape[1]):
    with torch.no_grad():
        y_hat, hidden= model(x_test[:,i].view(-1,1,26), hidden)

Y_max = 16.103195566
max1 = 16.1074
y_ans = y_hat.squeeze(0).numpy()*Y_max

pd.DataFrame(y_ans).to_csv("result1.csv")
print(y_ans)
