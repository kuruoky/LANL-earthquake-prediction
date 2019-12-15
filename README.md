LANL-Earthquake-Prediction

This project includes for multiple trials on extracting of the feature of the data as well as the modeling of the problem.

#------------------------------------------------------
In all of the .py files,

eng.py is the feature engineering of the training data and the testing data provided to us.
	eng.py is a must-run if you intend to run model.py, model1.py, model2.py and this takes a long time and a large space.

eng1.py is the new feature engineering that allows RNN network to incorporate more features, this takes less time given that you have run eng.py
	eng1 is a must-run if you intend to run model1.py.

model.py is the RNN solution of the problem which has the MAE of 2.05678432 on CV and 1.575 on LB

model1.py is the engineered version of the problem to incoporate more feature of the data into the RNN network that runs around 2.0355472984 on CV and 1.476 on LB score.

model2.py is the probablistic random forest solution of the question that includes the same amount of features as model1, the model runs around 2.2 on CV and 1.61 on LB.

output.py is the prediction process that gives the result after model.py has finished the training process

output1.py is the prediction process that gives the result after model1.py has finished the training process

tunning.py is a component of the LightGBM solution which runs a searching on possible combinations of the training specifications and save the best in range.

lgb.py is the version1 of the LightGBM solution of the problem that includes around 100 features of the given training data. The model gives 2.08... on intial commit using the default specifications on CV, and around 1.51 on LB. And after tunning, this lgb model gives 2.045 on CV and around 1.456 on LB which is a significant improve. And interestingly, the model does a 5-fold on the training dataset, and average on 5 results acquired from the training. If we submit only the result from the first fold, without averaging, the LB score goes up to 1.448.

lgb_andrew.py is the current version whereas our best LB score is acquired. The feature extraction is based on one of the remarkable kernel contributor who extracts over 2000 relevant features of the training data and shared them. (https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples)
Our modeling is based on around 900 of them selected based on our data exploration of the training set.
This version of the model gives around 2.02 on CV score and around 1.442 on intial commit to the LB. And further optimization of the features and bagging of the sets have improved greatly the score of the method to around 1.420.

#------------------------------------------------------
To run the the .pys above, one needs to build a folder named data and put all the training set and test set(extracted folder) within.

Special thanks to the kernel contributor: Andrew Lukyanenko, Scirpus and Oliver who inspires us on the engineering and manipulation of the data.

For more details on the model that is being implemented and the initial data exploration, please refer to our report enclosed.