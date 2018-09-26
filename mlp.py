import numpy as np
import csv
import timeit
# from multiprocessing.pool import ThreadPool
# pool = ThreadPool(processes=1)

X = []


# with open('foo.csv') as csvfile:
with open('train-num.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	next(readCSV)  # Skip header line
	for row in readCSV:
		a = list(map(float, row))
		X.append(a)


class Perceptron():
	"""docstring for Perceptron"""
	def __init__(self,num_inputs,optimizer,activation):
		super(Perceptron, self).__init__()
		
		self.num_inputs = num_inputs
		self.weights = np.random.rand(self.num_inputs)
		self.bias = np.random.rand(1)[0]

		if activation == "none":
			self.activation = 0
		elif activation == "sigmoid":
			self.activation = 1


		if optimizer == "GradientDescent":
			self.optimizer = 0 # gradient descent
			self.learning_rate = 0.5
		
		elif optimizer == "Adam":
			self.optimizer = 1 # Adam
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.epsilon = 1e-5
			self.m = np.asarray([0.0]*self.num_inputs)
			self.v = np.asarray([0.0]*self.num_inputs)
			self.learning_rate = 0.001
			self.m_b = 0.0
			self.v_b = 0.0
		
		self.timestep = 1
		self.batch_grad_w = np.zeros(self.num_inputs)
		self.batch_grad_b = 0.0
		self.batch_count = 0
		self.batch_size = 50

		

	def feedforward(self,inputs):

		output = np.sum(self.weights*inputs)+self.bias

		if self.activation == 1:
			output = self.sigmoid(output)

		return output

	
	def backprop(self,inputs,p_grad):

		inner_work = (np.sum(self.weights*np.asarray(inputs))+self.bias)
		if self.activation == 1:
			inner_work = self.sigmoid(inner_work)
			grad_b = p_grad*inner_work*(1-inner_work)
			grad_w = grad_b*np.asarray(inputs)
		
		elif self.activation == 0:
			grad_b = p_grad
			grad_w = grad_b*np.asarray(inputs)

		if self.optimizer == 1:
			self.m = (self.beta1 * self.m) + ((1-self.beta1)*grad_w)
			self.v = (self.beta2 * self.v) + ((1-self.beta2)*(grad_w**2))
			bc_m = (self.m)/(1-self.beta1**self.timestep)
			bc_v = (self.v)/(1-self.beta2**self.timestep)

			update_value_w = (bc_m)/(np.sqrt(bc_v)+self.epsilon)

			self.m_b = (self.beta1 * self.m_b) + ((1-self.beta1)*grad_b)
			self.v_b = (self.beta2 * self.v_b) + ((1-self.beta2)*(grad_b**2))
			bc_m = (self.m_b)/(1-self.beta1**self.timestep)
			bc_v = (self.v_b)/(1-self.beta2**self.timestep)
			update_value_b = (bc_m)/(np.sqrt(bc_v)+self.epsilon)

			

			if self.batch_count%self.batch_size == 0 and self.batch_size != 1:
			
				self.weights = self.weights - self.learning_rate*self.batch_grad_w
				self.bias = self.bias -  self.learning_rate*self.batch_grad_b
				
				self.batch_grad_w = np.zeros(self.num_inputs)
				self.batch_grad_b = 0.0

			elif self.batch_size == 1:
				self.weights = self.weights - self.learning_rate*update_value_w
				self.bias = self.bias -  self.learning_rate*update_value_b

			else:
				self.batch_grad_w += update_value_w
				self.batch_grad_b += update_value_b

		else:
			if self.batch_count%self.batch_size == 0 and self.batch_size != 1:
				
				self.weights = self.weights - self.learning_rate*self.batch_grad_w
				self.bias = self.bias -  self.learning_rate*self.batch_grad_b
				
				self.batch_grad_w = np.zeros(self.num_inputs)
				self.batch_grad_b = 0.0

			elif self.batch_size == 1:
				self.weights = self.weights - self.learning_rate*grad_w
				self.bias = self.bias -  self.learning_rate*grad_b

			else:
				self.batch_grad_w += grad_w
				self.batch_grad_b += grad_b

		self.batch_count += 1
		self.timestep+=1
		
		return p_grad*(self.weights)


	def sigmoid(self,inputs):

		return 1.0/(1.0 + np.exp(-inputs))


	def set_learning_rate(self,l_rate):

		self.learning_rate = l_rate

	def print_params(self):
		print("\tPrinting Unit Params:")
		print("\t",self.weights)
		print("\t",self.bias)

	def error(self,inputs,outputs):

		predicted = self.feedforward(inputs)

		return 0.5*((predicted - outputs)**2)





class Graph():

	def __init__(self,num_input,layer_sizes,optimizer,activations):

		print("Allocating memory for nodes in graph...")

		self.optimizer = optimizer
		self.activations = activations
		self.num_input = num_input
		self.layer_sizes = layer_sizes
		self.graph = [[Perceptron(num_input,optimizer,activations[0]) for _ in range(layer_sizes[0])]]

		for x in range(1,len(layer_sizes)):
			self.graph.append([Perceptron(layer_sizes[x-1],optimizer,activations[x]) for _ in range(layer_sizes[x])])

		print("Graph initialized.")
	def feedforward(self,inputs):

		input_holder = inputs
		layer_outputs = [input_holder]
		#  layer_outputs need to have input as index 0

		for layer in self.graph:
			layer_output = []
			for unit in layer:
				layer_output.append(unit.feedforward(input_holder))
			input_holder = np.asarray(layer_output)
			layer_outputs.append(layer_output)
			

		final_output= input_holder
		return final_output,layer_outputs


	def backprop(self,layer_outputs,expected):

		p_grad = [(layer_outputs[-1][0] - expected)]*len(self.graph[-1])
		for i in range(len(self.graph)):
			k = 0
			temp = np.asarray([0.0]*len(layer_outputs[len(self.graph)-i-1]))
			
			for unit in self.graph[len(self.graph)-i-1]:
				# async_result = pool.apply_async(unit.backprop,[layer_outputs[len(self.graph)-i-1],p_grad[k]])
				temp += unit.backprop(layer_outputs[len(self.graph)-i-1],p_grad[k])#async_result.get()
				k+=1
			p_grad = temp


	def print_params(self):

		print("Printing Graph Params")
		for layer in self.graph:
			for unit in layer:
				unit.print_params()


	def dump_params(self,file="graphdump.txt"):
		with open(file, 'w') as f:
			f.write(str(self.num_input))
			f.write("\n")
			for x in self.layer_sizes:
				f.write(str(x))
			f.write("\n")
			f.write(str(self.activations))
			f.write("\n")
			f.write(str(self.optimizer))
			f.write("\n")
			
			for layer in self.graph:
				for unit in layer:
					f.write(str(unit.weights))
					f.write("\n")
					f.write(str(unit.bias))
					f.write("\n")


	# def loadfrom(file):
	# 	with open(file, 'r') as f:


	def error(self,inputs,outputs):

		final_pred,l = self.feedforward(inputs)
		return 0.5*((final_pred - outputs)**2)

if __name__ == "__main__":

	start = timeit.default_timer()

	#  num_inputs , layer_sizes
	activations=["none","sigmoid"]
	g = Graph(15,[ 1,      1     ],"Adam",activations)

	epochs = 10

	# x = X[0][0]
	# y = X[0][1]
	# o = X[0][2]
	
	total = len(X)
	for e in range(epochs):
		i=0
		for row in X:
			print(int((i/total)*100),"%", end="\r", flush=True)
			# print(i,"%", end="\r", flush=True)
			f,l = g.feedforward(row[1:-1])
			g.backprop(l,row[-1])
			i+=1
			# if i==20000:break
		print("Epoch no.",e+1,"done")

	stop = timeit.default_timer() - start
	suff = " secs"
	if stop > 60.0:
		stop = stop / 60.0
		suff = " mins"
		if stop > 60.0:
			stop = stop / 60.0
			suff = " hrs"			


	print('Time: ', str(stop)+suff) 

	g.dump_params("graphdump.txt")

	i=0
	for row in X:
		f,l = g.feedforward(row[1:-1])
		print(f,row[-1])
		i+=1
		if i==10:exit()


	# print("Total Error:")
	# err = 0
	# for [x,y,o] in X:
	# 	err += g.error([x,y],o)
	# print(err)
