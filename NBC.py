import numpy as np


def splitDataset(dataset, splitRatio):
    # Training set size
    trainSize = int(dataset.shape[0] * splitRatio)

    # List of randomly chosen indicies
    indices = np.random.permutation(dataset.shape[0])

    # Split indicies for training and test set by trainSize
    training_idx, test_idx = indices[:trainSize], indices[trainSize:]

    # Create training and test sets by indicies
    training, test = dataset[training_idx, :], dataset[test_idx, :]

    return training, test



def separateByClass(dataset, classes):
    # Here we limit our classes to 0 and 1
    # You need to generalize this for arbitrary number of classes
    sep_classes = {}
    for i in classes:
        sep_classes[i] = dataset[np.where(dataset[:, -1] == i), :]
    return sep_classes


# In[20]:


def summarize(dataset):
    # Calculate means and standart deviations with one degree of freedom for each attribute
    # We do it by column which is axis 1
    # Also we remove last elements (guess why?)
    means = dataset.mean(axis=1)[0][:-1]
    stds = dataset.std(axis=1, ddof=1)[0][:-1]

    # Think what we do here?

    return means, stds


def summarizeByClass(dataset, classes):
    # Divide dataset by class and summarize it
    separated = separateByClass(dataset, classes)

    summaries = {}

    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)

    return summaries


def calculateProbability(x, mean, stdev):
    # Calculate probability by x, mean and std
    # 1/(sqrt(2pi)*std)*exp(-(x-mean)^2/(2std^2))
    return np.prod(np.exp(-(x - mean) ** 2 / (2 * stdev ** 2)) / (np.sqrt(2 * np.pi) * stdev))


def calculateClassProbabilities(summaries, inputVector):
    # Calculate probabilities for input vector from test set
    probabilities = {}

    for classValue, classSummaries in summaries.items():
        means = classSummaries[0]
        stds = classSummaries[1]

        # Calculate corresonding probabilities and multiply them
        probabilities[classValue] = calculateProbability(inputVector[:-1], means, stds)

    return probabilities


def predict(summaries, inputVector):
    # Calculate probabilities
    probabilities = calculateClassProbabilities(summaries, inputVector)

    # Init values of probability and label
    bestLabel, bestProb = None, -1

    # Check probability of which class is better
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel

def getPredictions(summaries, testSet):
    # For each probability find optimal labels
    # Передлать на Numpy
    predictions = []

    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions


def getAccuracy(testSet, predictions):
    # Check accuracy
    correct = 0

    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():

    # Set split ratio
    splitRatio = 0.67

    # Load dataset and return numpy array
    dataset = np.genfromtxt('/home/vladimir/PycharmProjects/dataroot/data_NBC.csv', delimiter=',')

    classes = [0, 1]

    # Split dataset
    trainingSet, testSet = splitDataset(dataset, splitRatio)

    # Log row amounts
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))

    # Prepare model
    summaries = summarizeByClass(trainingSet, classes)

    # Test model
    predictions = getPredictions(summaries, testSet)

    accuracy = getAccuracy(testSet, predictions)

    print('Accuracy: {0}%'.format(accuracy))


main()

