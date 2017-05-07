# -*- coding: utf-8 -*-
"""
Created on Sat May  6 23:08:44 2017

@author: rafael.sandroni
"""
from random import seed
from random import random
from math import exp

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
#usando a função de transferência sigmoide, é calculada sua derivada
def transferDerivative(output):
    return output * (1.0 - output)
    

def backwardPropagateError(network, expectedOutput):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expectedOutput[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transferDerivative(neuron['output'])


def updateWeights(network, row, learningRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i -1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learningRate * neuron['delta']

def trainNetwork(network, train, learningRate, nEpoch, nOutputs):
    
    for epoch in range(nEpoch):
        sumError = 0
        for row in train:
            outputs = forwardPropagate(network, row)
            expectedOutput = [0 for i in range(nOutputs)]
            expectedOutput[row[-1]] = 1
            sumError += sum([( expectedOutput[i] - outputs[i])**2 for i in range(len(expectedOutput))])
            backwardPropagateError(network, expectedOutput)
            updateWeights(network, row, learningRate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sumError))

def predict(network, row):
    outputs = forwardPropagate(network, row)
    return outputs.index(max(outputs))
    
    
# test training backprop algorithm

seed(1)

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


for layer in network:
    print(layer)
    #for neuron in layer:        
    #    print(neuron)

for row in dataset:
    prediction = predict(network, row)
    print("Expected=%d, Got=%d" % (row[-1], prediction))
