# BazaarPredictionModel
A prediction model that is meant for predicting Hypixel Skyblock Bazaar Prices

Python Version: 3.8.6 or above

Dependencies: Tensorflow 2.0 or above , Numpy and Matplotlib
  
Extract BazaarData.zip, it contains the training data and the item list. After, run Main.py and the model will start to train. It autosaves the model every 25 batches but that can be changed in the code.

The data has been gathered from [stonks.gg](https://stonks.gg/). It represents the data from 4/4/2020 00:00:00 to 4/12/2020 00:00:00 where 1 datapoint represents the averege of 6 hours worth of data. Each json in the database contains an 2d array where the first dimension has the length of 2, index 0 being the buy prices and index 1 being the sell prices.
