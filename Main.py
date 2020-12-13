import json
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BUY_INDEX = 0
SELL_INDEX = 1

InputDays = 5 # 5 Days for input
OutputDays = 5 # Has to predict 5 days
DatapointsPerDay = 4 # Each data point represents 6 hours so 4 in a day

TotalDataPoints = 976

def JsonToNumpy(Path,NeedsSplit):
    if (NeedsSplit):
        Data = open(Path,"r").read().split("\n")
        Arr1 = [np.array(arr) for arr in json.loads(Data[0])]
        Arr2 = [np.array(arr) for arr in json.loads(Data[1])]
        return np.array([Arr1,Arr2])
    else:
        Arr = [np.array(arr) for arr in json.loads(open(Path,"r").read())]
        return np.array(Arr)

ItemList = JsonToNumpy("ItemList.json",False)

Dataset = np.zeros((ItemList.size,2,TotalDataPoints-1)) #There are normally 976 data points but since this represents the derivatives it's -1 smaller

def GetDerivativeOfArray(Arr):
    Derivative = np.zeros(Arr.size-1)
    for i in range(Arr.size-1):
        Derivative[i] = Arr[i+1] - Arr[i]
    return Derivative

# Loads the data set
for i in range(ItemList.size):
    ItemData = JsonToNumpy("Dataset/"+ItemList[i].replace(":","")+".json",True)
    Deriatives = np.array([GetDerivativeOfArray(ItemData[BUY_INDEX]),GetDerivativeOfArray(ItemData[SELL_INDEX])])
    Dataset[i] = Deriatives

DataPointsInExample = InputDays * DatapointsPerDay + OutputDays * DatapointsPerDay - 1 # -1 because it's gonna take the derivative of the graph

TrainingExamples = np.zeros((TotalDataPoints-1 - DataPointsInExample,ItemList.size,2,DataPointsInExample))

# Splits into TrainingExamples
# Total of 936 batcthes
# 39 datapoints per batch
for i in range(ItemList.size):
    for j in range(TotalDataPoints-1 - DataPointsInExample):
        StartIndex = j
        EndIndex = j + DataPointsInExample
        TrainingExamples[j][i] = Dataset[i][:,StartIndex:EndIndex]

Batches = tf.data.Dataset.from_tensor_slices(TrainingExamples).shuffle(TotalDataPoints-1 - DataPointsInExample).batch(64)

# Builds the model
def BuildModel():
    Model = tf.keras.Sequential()
    Model.add(tf.keras.layers.Input(shape=(189,2,20)))

    Model.add(tf.keras.layers.Flatten())

    Model.add(tf.keras.layers.Dense(3000,activation="relu"))
    Model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    Model.add(tf.keras.layers.Dropout(0.5))

    Model.add(tf.keras.layers.Dense(1000,activation="relu"))
    Model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    Model.add(tf.keras.layers.Dropout(0.5))

    Model.add(tf.keras.layers.Dense(3000,activation="relu"))
    Model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    Model.add(tf.keras.layers.Dropout(0.5))

    Model.add(tf.keras.layers.Dense(7182,activation="tanh"))

    Model.add(tf.keras.layers.Reshape((189,2,19)))

    return Model

Model = BuildModel()
#Model = tf.keras.models.load_model("BazaarAI")
Model.summary()

Optimizer = tf.optimizers.Adadelta(learning_rate=0.006)

# Train function
def TrainModel():
    # Value used for autosaving
    Runs = 0
    EpochLoss = 0
    while True:
        print("Epoch Loss - " + str(EpochLoss))
        EpochLoss = 0
        for Example in Batches:
            Runs += 1

            Example = Example.numpy()

            # Normalizez the input
            for b in range(Example.shape[0]):
                for i in range(ItemList.size):
                    MaximumBuyValue = 0
                    MaximumSellValue = 0
                    for j in range(39):
                        if (MaximumBuyValue < abs(Example[b][i][BUY_INDEX][j])):
                            MaximumBuyValue = abs(Example[b][i][BUY_INDEX][j])
                        if (MaximumSellValue < abs(Example[b][i][SELL_INDEX][j])):
                            MaximumSellValue = abs(Example[b][i][SELL_INDEX][j])
                    Example[b][i][BUY_INDEX] /= MaximumBuyValue
                    Example[b][i][SELL_INDEX] /= MaximumSellValue

            with tf.GradientTape() as Tape:
                Input = Example[:,:,:,:20]
                ExpectedPrediction = Example[:,:,:,20:]
                
                Prediction = Model(Input)

                # This is for human discrimination of some result generated
                # Right now this shows the results for item 13 but any item can work
                if (Runs == 25):
                    plt.plot(Example[0][13][0])
                    plt.plot(np.concatenate([Example[0][13][0][:20],Prediction[0][13][0]]))
                    plt.savefig("out.png")
                    plt.clf()
                
                Loss = tf.losses.mean_absolute_error(ExpectedPrediction,Prediction)
                EpochLoss += np.mean(Loss)
                print("Loss - " + str(np.mean(Loss)))

                Gradients = Tape.gradient(Loss,Model.trainable_variables)
                Optimizer.apply_gradients(zip(Gradients,Model.trainable_variables))

            if (Runs == 25):
                Model.save("BazaarAI")
                Runs = 0

TrainModel()