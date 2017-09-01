# -*- coding: utf-8 -*-
"""
Created on Sat May  6 23:08:44 2017

@author: rafael.sandroni
"""
from random import seed
from random import random
from math import exp
from random import randrange
from csv import reader

#functions to load data from csv file

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluateAlgorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


#funções para rede neural


#Inicializa a rede neural
def initializeNetwork(n_inputs, n_hidden, n_outputs):
    #Um array em forma de dicionário (network)
    #Com todos os elementos da rede neural
    network = list()
    
    #Inicializa os pesos da camada oculta(hidden), criando uma matriz com valores aleatóricos 
        
    #random data for i in range(n_inputs + 1) <= create a vector, data of each input
    #each vector for i in range(n_hidden) <= create a matrix of vectors, a row with data of each weights input
    
    #create inputs, for neurons
    #para cada dado de entrada (inputt + bias), insere os pesos+bias e conecta ao neuronio da camada oculpa (hidden)
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    #insere a matriz de pesos no dicionário da rede neural
    network.append(hidden_layer)
    #Para cada dado na camada oculta, insere os pesos+bias para suas conexões com a camada de saída
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    #insere a matriz de pesos no dicionario da rede neural
    network.append(output_layer)
    
    return network

def activateFunction(weights, inputs):
    #atribui o ultima index do vetor de pesos, que contêm o valor do bias
    activation = weights[-1]
    #para cada peso do vetor, 
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
        
    return activation

# Transfere a função de ativação, usando a função sigmoid

"""
A função de ativação sigmoid é semelhante a forma de S, e também é chamada de função logística.
Pode coletar qualquer valor de entrada e produzir um número entre 0 e 1 em uma curva S. 
É também uma função da qual podemos calcular facilmente a derivada (inclinação) que 
é necessário na fase de retropropagação do erro.
activation = sum(weight_i * input_i) + bias
"""
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


#execução da função de propagação para cada linha do dataset, com a rede neural
def forwardPropagate(network, row):
    #atribui o dado (cada linha do dataset) da camada de entrada
    inputs = row
    #para cada camada da rede neural
    for layer in network:
        #inicializa o vetor com a saida da camada atual
        new_inputs = []
        #para cada neuronio da camada atual
        for neuron in layer:
            #executa a função de ativação, calculando os pesos e entradas do neuronio atual
            activation = activateFunction(neuron['weights'], inputs)
            #executa a tranferência da função de ativação, e atribui para o vetor para saida do neuronio atual
            neuron['output'] = transfer(activation)
            #acumula o dado em um vetor de saida para o neuronio atual
            new_inputs.append(neuron['output'])
            
        #atribui a saida da camada atual, para ser usada como entrada na próxima camada
        inputs = new_inputs
    
    #vetor com o dado de saida da ultima camada da rede neural ( camada de saida )
    return inputs
    
#Dado um valor de saída de um neurônio, é calculado a sua inclinação.
#usando a derivada da função de transferência sigmoide
def transferDerivative(output):
    return output * (1.0 - output)
    
"""
metodo de propagação do erro
roda de forma decrescente todas as camadas da rede neural
extrai a diferença entre o esperado e o resultado da camada atual
calcula a taxa de erro para propagar nas camadas anteriores

@param network dict (dicionário)
@param expectedOutput vetor
"""
def backwardPropagateError(network, expectedOutput):
    #para cada camada na rede neural, varre de forma decrescente, da ultima para a primeira
    for i in reversed(range(len(network))):
        #camada atual
        layer = network[i]
        #inicializa list
        errors = list()
        #se não for a última camada (output)
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    """
                    error = (weight_k * error_j) * transfer_derivative(output)
                    Where error_j is the error signal from the jth neuron in the output layer, 
                    weight_k is the weight that connects the kth neuron to the current neuron and 
                    output is the output for the current neuron.
                    """
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
                
        #exclusivo para a ultima camada
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expectedOutput[j] - neuron['output'])
                
        #para cada neuronio da camada atual
        for j in range(len(layer)):
            neuron = layer[j]
            #define o delta, dado um valor de saida, produto do erro e saida
            #delta é a taxa de erro calculada para cada neuronio
            neuron['delta'] = errors[j] * transferDerivative(neuron['output'])


#atualizacao dos pesos
def updateWeights(network, row, learningRate):
    #para cada camada da rede neural
    for i in range(len(network)):
        #get all row
        inputs = row[:-1]
        if i != 0:
            #for others layers, except the first, get output for each previous neuron
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            #for each input
            for j in range(len(inputs)):
                #calcule weights with learningRate, delta weights, and input
                neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
            #bias
            neuron['weights'][-1] += learningRate * neuron['delta']


#training network
def trainNetwork(network, train, learningRate, nEpoch, nOutputs):
    #conforme numero de epocas, faz o trainamento da rede neural
    for epoch in range(nEpoch):
        sumError = 0
        #para cada linha do dataset de treinamento
        for row in train:
            #calcula o valor de saida da rede neural, obtido na ultima camada
            outputs = forwardPropagate(network, row)
            #o valor resultante esperado
            expectedOutput = [0 for i in range(nOutputs)]
            
            expectedOutput[row[-1]] = 1
            #row[-1] = 1
            #verifica a taxa de erro
            sumError += sum([( expectedOutput[i] - outputs[i])**2 for i in range(len(expectedOutput))])
            #propagação do erro nas camadas, de forma decrescente
            backwardPropagateError(network, expectedOutput)
            #atualização dos pesos, para ser usada na proxima epoca
            updateWeights(network, row, learningRate)
            
        #exibe métricas sobre o comportamento da rede
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sumError))

#precision
def predict(network, row):
    outputs = forwardPropagate(network, row)
    return outputs.index(max(outputs))
    
# Backpropagation Algorithm
def backPropagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initializeNetwork(n_inputs, n_hidden, n_outputs)
	trainNetwork(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)
 
# test training backprop algorithm

seed(1)
"""
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

n_inputs = len(dataset[0]) -1
n_outputs = len(set([row[-1] for row in dataset]))

#create a neural network
network = initializeNetwork(n_inputs,1,n_outputs)
trainNetwork(network, dataset, 0.5, 20, n_outputs)

for row in dataset:
    prediction = predict(network, row)
    print("Expected=%d, Got=%d" % (row[-1], prediction))
"""

# load and prepare data
filename = 'seeds_dataset3.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 2
l_rate = 0.3
n_epoch = 500
n_hidden = 6
scores = evaluateAlgorithm(dataset, backPropagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
