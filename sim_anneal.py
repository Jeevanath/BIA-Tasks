import numpy as np
import math
from matplotlib import pyplot as plt
from  matplotlib.animation import FuncAnimation
from matplotlib import cm
from numpy.random import normal, uniform
from numpy import exp, cos, pi
from math import sqrt

def sphere_function(x):
	sum = 0
	for i in x:
		sum += i**2
	return sum

def schwefel_function(x):
	sum = 0
	for i in x:
		sum += i * np.sin(math.sqrt(abs(i)))
	return 418.9829*2 - sum


def rastrigin_function(x):
	d = len(x)
	sum = 0
	for i in x:
		sum += i**2 - 10*(math.cos(2*3.14*i))

	return 10 + sum



def rosenbrock_function(x):
	d = len(x)
	sum = 0
	for i in range(0, len(x)-1):
		sum += (100*(x[i+1] - (x[i])**2))**2 + ((x[i] - 1)**2)
	return sum


def ackley_function(x):
	d=len(x)

	sum1 = 0
	sum2 = 0
	term1 = 0
	term2 = 0

	for i in x:
		sum1 +=  (i ** 2)
		sum2 +=  cos(2 * pi * i)
	term1 = -20 * exp(-0.2 * sqrt(sum1/d))
	term2 = -exp(sum2/d)
	y = term1 + term2 + 20 + exp(1)
	return y

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
	


def sim_anneal(lmin, lmax, fn):
	T0 = 100
	T_min = 0.5
	alpha = 0.95	

	trace = []
	for _ in range(20):
		T = T0
		while T > T_min:
			neighbours = [normal((lmin, lmax) , 0.5) for number in range (100)] 
			for n in range(0, 100):
				for m in range(0, 2):
					 neighbours[n][m]= clamp(neighbours[n][m], lmin, lmax)

			evaluations = [fn.fname(x) for x in neighbours]
			mini_index = 0
			for i,v in enumerate(evaluations):
				if v < evaluations[mini_index]:
					mini_index=i
				else:
					r = uniform(lmin, lmax)
					diff = v - evaluations[mini_index]
					E = exp(-diff/T)
					if r < E:
						mini_index = i
			T = T * alpha
		lmin, lmax = [*neighbours[mini_index]]
		trace.append([*neighbours[mini_index], evaluations[mini_index]])

			
	return trace




def make_surface(func, xmin, xmax, ymin, ymax, step):
	X = np.arange(xmin, xmax, step)
	Y = np.arange(ymin, ymax, step)
	Z = []

	for x in X:
		temparr = []
		for y in Y:
			temparr.append(func([x,y]))
		Z.append(np.array(temparr))
	X, Y = np.meshgrid(X, Y)
	Z = np.array(Z)
	return (Y,X,Z)
	

def update(frame, data, scat, ax):

	scat[0].remove()
	scat[0] = ax[0].scatter(data[frame][0], data[frame][1], data[frame][2], c='red')
	



def init_frame():
	print("Repeat")




def render_animation(X, Y, Z, data, fn): 
	fig = plt.figure()
	ax = plt.axes(projection ='3d')

	if fn:
		ax.set_title(fn.name)
	# Creating plot
	surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm, linewidth = 0, antialiased = False, alpha = 0.6)

	# scat = ax.scatter([x[0] for x in data], [y[1] for y in data],[z[2] for z in data], c='red')
	scat  = ax.scatter([data[0][0]], [data[0][1]],[data[0][2]], c='red')
	ani = FuncAnimation(fig, func = update, frames =  len(data) , fargs=(data, [scat],[ax]), init_func = init_frame)
	plt.show()
	


	


class Function:
	def __init__(self, name, fname, xmin, xmax, ymin, ymax, step):
		self.name = name
		self.fname = fname
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.step = step



sphere = Function("Sphere function",sphere_function, -5.12, 5.12, -5.12, 5.12, 0.1)
schwefel = Function("Schwefel function",schwefel_function, -500, 500, -500, 500, 1)
rastrigin = Function("Rastrigin function", rastrigin_function, -5.12, 5.12, -5.12, 5.12, 0.01)
rosenbrock = Function("Rosenbrock function", rosenbrock_function, -10, 10, -10, 10, 0.01)
ackley = Function("Ackley function", ackley_function, -32.768, 32.768, -32.768, 32.768, 0.1)






# functions_list = [sphere, schwefel, rastrigin, rosenbrock, ackley]
functions_list = [sphere]



for fn in functions_list:
	data = []
	X, Y, Z = make_surface(fn.fname, fn.xmin, fn.xmax, fn.ymin, fn.ymax, fn.step)
	# data=blind_search(fn.xmin, fn.xmax)


	data = sim_anneal(fn.xmin, fn.xmax, fn)
	print(data)
	render_animation(X, Y, Z, data, fn)

				
	
	
	
	

	






















