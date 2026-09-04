import math, random, statistics
import numpy as np
from typing import List, Tuple, Callable, Optional

class Engine:
	parameters = None;
	optimizer = lambda : None;
	def zerograd(self): [x.zerograd() for x in Engine.parameters]; return self;


class Variable(Engine):
	def __init__(self, data=np.float32(), fn=None): self.data = data; self.type = Variable; self.id=None; self.grad=0; self.fn=fn;
	@staticmethod
	def uniform(lower=0,upper=1.0,fn=None): return Variable(data=np.random.uniform(lower,upper),fn=fn);
	@staticmethod
	def normal(mean=0, std=1.0, fn=None): return Variable(data=np.random.normal(mean, std), fn=fn)
	@staticmethod
	def randint(low=0, high=1, fn=None): return Variable(data=np.random.randint(low, high), fn=fn)

	def __str__(self): return f"{self.id} = {self.__repr__()}" if self.id!=None else f"{self.__repr__()}" 
	def __clr__(self, x:str): return f"\033[92m+{x}\033[0m" if self.data>0 else f"\033[91m{x}\033[0m"
	def __repr__(self,precision=5): return self.__clr__(f"{self.data:.{precision}f}")

	def __add__(x, y): return Function(x, y, op=lambda _:_[0].data+_[1].data, grad=[lambda _:1, lambda _:1])()
	def __mul__(x, y): return Function(x, y, op=lambda _:_[0].data*_[1].data, grad=[lambda _:_[1].data, lambda _:_[0].data])()
	def __pow__(x, y): return Function(x, y, op=lambda _:_[0].data**_[1].data, grad=[lambda _:_[1].data*_[0].data**(_[1].data-1), lambda _: 1])()
	def __neg__(x): return x * (-1);
	def __radd__(x, y): return x + y 
	def __rmul__(x, y): return x * y 
	def __sub__(x, y): return x + (-y)
	def __rsub__(x, y): return (-x) + y 
	def __truediv__(x, y): return x * (float(y)**-1);
	def __rtruediv__(x, y): return y * (float(x)**-1);

	def log(x): return Function(x, op=lambda _:np.log(_[0].data), grad=[lambda _:1/_[0].data])()
	def relu(x): return Function(x, op=lambda _:max(0,_[0].data), grad=[lambda _:1 if (_[-1].data>0) else 0])()
	def exp(x): return Function(x, op=lambda _:np.exp(_[0].data), grad=[lambda _:_[-1].data])()
	def sqrt(x): return Function(x, op=lambda _:np.sqrt(_[0].data), grad=[lambda _:.5/_[-1].data])()
	def tanh(x): return Function(x, op=lambda _: np.tanh(_[0].data), grad=[lambda _: 1 - _[-1].data**2])()
	def abs(x): return Function(x, op=lambda _: np.abs(_[0].data), grad=[lambda _: 1 if _[0].data > 0 else -1])()
	def sigmoid(x): return Function(x, op=lambda _:1/(1+np.exp(-_[0].data)), grad=[lambda _:_[-1].data*(1-_[-1].data)])()

	def backward(self): return self.fn.backward() if self.fn else (_ for _ in ()).throw(Exception("Function not set"))
	def zerograd(self): self.grad=0; return self;

class Function(Engine):
	def __init__(self, *args, op=lambda: None, grad=lambda: None): 
		self.op = op; 
		self.grad = grad;
		self.args = [_ if isinstance(_,Variable) else Variable(_) for _ in args]+[Variable(fn=self)]; 
	def backward(self): 
		topo = []; visited = set();
		def build_topo(f):
			if (f==None): return;
			if f not in visited:
				visited.add(f)
				for _ in f.args[:-1]: build_topo(_.fn)
				topo.append(f)
		build_topo(self)
		self.args[-1].grad=1
		for f in topo[::-1]: 
			for _,grad in zip(f.args[:-1],f.grad): _.grad+=grad(f.args) * f.args[-1].grad; 
	def __call__(self): self.args[-1].data=self.op(self.args); return self.args[-1]

class Tensor(Engine):
	def __init__(self, *shape, data=None): 
		if shape and isinstance(shape[0],List): self.data=np.vectorize(Variable)(np.array(shape[0])); return;
		if shape and isinstance(shape[0],np.ndarray): self.data=shape[0]; return;
		if isinstance(data,List): data=np.array(data);
		self.data = data.reshape(shape) if isinstance(data,np.ndarray) else np.array([Variable()  for _ in range(math.prod(shape))]).reshape(shape)
	@staticmethod
	def fill(shape, value): return Tensor(*shape, data=np.array([Variable(value) for _ in range(math.prod(shape))]))
	@staticmethod
	def ones(*shape):  return Tensor(*shape, data=np.array([Variable(1) for _ in range(math.prod(shape))]))
	@staticmethod
	def zeros(*shape): return Tensor(*shape, data=np.array([Variable(0) for _ in range(math.prod(shape))]))
	@staticmethod
	def eye(n): return Tensor(n, n, data=np.array([Variable(1 if i == j else 0) for i in range(n) for j in range(n)]))
	@staticmethod
	def uniform(*shape,lower=-1,upper=1): return Tensor(*shape, data=np.array([Variable.uniform(lower,upper) for _ in range(math.prod(shape))]))
	@staticmethod
	def normal(*shape,mean=0,sd=1): return Tensor(*shape, data=np.array([Variable.normal(mean,sd) for _ in range(math.prod(shape))]))
	@staticmethod
	def arange(*shape): return Tensor(*shape, data=np.array([Variable(_+1) for _ in range(math.prod(shape))]))
	@staticmethod
	def randint(*shape,low=0,high=1): return Tensor(*shape, data=np.array([Variable(np.random.randint(low,high+1)) for _ in range(math.prod(shape))]))
	@staticmethod
	def cast(data): return Tensor(*data.shape, data=data) if isinstance(data,np.ndarray) else Tensor(data=data)

	@property
	def shape(self): return self.data.shape
	@property
	def size(self): return len(self.data)
	@property
	def rank(self): return len(self.shape)
	@property
	def grad(self): return Tensor(np.array([Variable(x.grad) for x in self.data.flat]).reshape(self.data.shape))

	def reshape(self, *shape): self.data=self.data.reshape(shape); return self; 
	def transpose(self): self.data=np.transpose(self.data); self.reshape(self.data.shape)
	def flatten(self): self.data=self.data.flatten(); self.reshape=(math.prod(self.shape)); return self;
	def concat(self, other): return Tensor(np.concatenate((self.data, other.data)));
	
	def __add__(a, b): return Tensor(a.data + b.data)
	def __sub__(a, b): return Tensor(a.data - b.data)
	def __mul__(a, b): return Tensor(a.data * b.data)
	def __truediv__(a, b): return Tensor(a.data / b.data)
	def __neg__(a): return Tensor(-a.data)
	def __pow__(a, b): return Tensor(a.data ** b)

	def __radd__(a, b): return Tensor(b + a.data)
	def __rsub__(a, b): return Tensor(b - a.data)
	def __rmul__(a, b): return Tensor(b * a.data)
	def __rtruediv__(a, b): return Tensor(b / a.data)
	def __rpow__(a, b): return Tensor(b ** a.data)
	def __isub__(a, b): return Tensor(a.data - b.data)
	def __iadd__(a, b): return Tensor(a.data + b.data)

	def reduce(self, fn, axis=None, keepdims=False): return Tensor(np.array([fn(axis=axis, keepdims=keepdims)]))
	def sum(self, axis=None, keepdims=False): return self.reduce(self.data.sum, axis, keepdims)
	def mean(self, axis=None, keepdims=False): return self.reduce(self.data.mean, axis, keepdims)
	def max(self, axis=None, keepdims=False):  return self.reduce(self.data.max, axis, keepdims)
	def min(self, axis=None, keepdims=False):  return self.reduce(self.data.min, axis, keepdims)

	def dot(a, b): return Tensor(a.data.dot(b.data))
	def __matmul__(a, b): return Tensor(a.data @ b.data)
	def conv1d(x, kernel): 
		return Tensor(np.array(
			[ (x.data[i:i+len(kernel)] * kernel.data).sum() 
			for i in range(x.shape[0] - kernel.shape[0] + 1) 
		]))
	def conv2d(x, kernel):
		return Tensor(np.array([
			[(x.data[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel.data).sum()
			 for j in range(x.shape[1] - kernel.shape[1] + 1)]
			for i in range(x.shape[0] - kernel.shape[0] + 1)
		]))
	def batchnorm(x, eps=1e-5):
		mean = x.mean(axis=0, keepdims=True)
		var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
		return (x - mean) / (var + eps) ** 0.5
	def layernorm(x, eps=1e-5):
		mean = x.mean(axis=-1, keepdims=True)
		var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
		return (x - mean) / (var + eps) ** 0.5
	def softmax(x, axis=-1): e = (x - x.max(axis, keepdims=True)).exp(); return e / e.sum(axis, keepdims=True)


	def unary(x, fn): return Tensor(np.array(list(map(fn, x.data.flat))).reshape(x.data.shape))
	def log(x): return x.unary(lambda x: x.log())
	def exp(x): return x.unary(lambda x: x.exp())
	def relu(x): return x.unary(lambda x: x.relu())
	def sqrt(x): return x.unary(lambda x: x.sqrt())
	def abs(x): return x.unary(lambda x: x.abs())
	def tanh(x): return x.unary(lambda x: x.tanh())
	def sigmoid(x): return x.unary(lambda x: x.sigmoid())

	def __setitem__(self, i, v): self.data[i]=v; return v;
	def __getitem__(self, _): return Tensor.cast(self.data[[int(x.data) for x in _.data.flatten()]]) if isinstance(_,Tensor) else self.data[_]
	def __str__(self): return self.data.__str__()
	def __repr__(self): return self.data.__repr__();
		
	def backward(self): [v.fn.backward() for v in self.data.flat if v.fn]; return self;
	def forward(self,layers): return layers[0](self).forward(layers[1:]) if layers else self
	def optimize(self): Engine.optimizer(); return self;

	
	def l1(self,l1_reg_strength=1e-4): return self + l1_reg_strength * (Engine.parameters.abs()).sum();
	def l2(self,l2_reg_strength=1e-4): return self + (l2_reg_strength/2) * (Engine.parameters ** 2).sum();
	def mse(p, y): return ((p-y)**2).mean()
	def mae(p, y): return (p-y).abs().mean()
	def huber(p, y, delta=1.0): e = (p-y).abs(); return ((e**2)/2).where(e <= delta, delta*(e-delta/2)).mean()
	def binaryCrossEntropy(p, y): p.flatten(); return -((y*p.log() + (1-y)*(1-p).log()).mean())
	def categoricalCrossEntropy(p, y): return -(y*p.log()).sum(axis=-1).mean()
	def sparseCategoricalCrossEntropy(p, y): return -p[range(y.shape[0]), y].log().mean()
		
class Layer(Engine):
	def __init__(self,*weights,forward,act): 
		self.weights = weights; self.forward=forward; self.act=act; 
		weights = np.concatenate([w.data.flat for w in self.weights]) if self.weights else np.array([])
		Engine.parameters = Engine.parameters.concat(Tensor(weights)) if Engine.parameters else Tensor(weights)
	@staticmethod
	def Linear(n,m,act=None): 
		return Layer(Tensor.normal(n,m),Tensor.normal(m), 
			   forward=lambda x,w:(x.dot(w[0]))+w[1],act=act)
	@staticmethod
	def Conv1d(cin, cout, k, act=None):
		return Layer( Tensor.normal(cout, cin, k), Tensor.normal(cout), 
			   forward=lambda x, w: x.conv1d(w[0]) + w[1], act=act)
	@staticmethod
	def Conv2d(kh, kw, act=None):
		return Layer( Tensor.normal(kh, kw), forward=lambda x, p: x.conv2d(p[0]), act=act)
	@staticmethod
	def BatchNorm(n, act=None):
		return Layer(
			Tensor.ones(n), Tensor.zeros(n),
			forward=lambda x, p: (x - x.mean(0, True)) / (x.var(0, True) + 1e-5) ** 0.5 * p[0] + p[1],
			act=act
		)
	@staticmethod
	def LayerNorm(n, act=None):
		return Layer(
			Tensor.ones(n), Tensor.zeros(n),
			forward=lambda x, p: (x - x.mean(-1, True)) / (x.var(-1, True) + 1e-5) ** 0.5 * p[0] + p[1],
			act=act
		)
	def __call__(self,x): return self.act(self.forward(x,self.weights)) if self.act!=None else self.forward(x,self.weights);

class Optimizer(Engine):
	def __init__(self,args,step=lambda self:None,update={}): 
		self.step=step;
		self.update=update;
		for k,v in args.items(): setattr(self, k, v)
	def __call__(self):
		for k,v in self.update.items(): setattr(self, k, v(self))
		delta = self.step(self)
		for x,self in zip(Engine.parameters.data, delta.data): x.data -= self.data;
	@staticmethod
	def SGD(lr=1e-3, momentum=0.9):
		return Optimizer(
			{'lr': lr, 'momentum': momentum, 'velocity': Tensor.zeros(Engine.parameters.size)},
			update={ 'velocity': lambda self: self.momentum * self.velocity + Engine.parameters.grad },
			step=lambda self: self.lr * self.velocity
		)
	@staticmethod
	def RMSprop(lr=1e-3, beta=0.9, eps=1e-8):
		return Optimizer({'lr':lr, 'beta':beta, 'eps':eps, 'adaptive_lr':Tensor.zeros(Engine.parameters.size)},
			update={'adaptive_lr':lambda self:self.beta*self.adaptive_lr + (1-self.beta) * Engine.parameters.grad**2},
			step=lambda self:self.lr * Engine.parameters.grad / (self.adaptive_lr + self.eps)
		)
	@staticmethod
	def Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		return Optimizer(
			{'lr': lr, 'beta1': beta1, 'beta2': beta2, 'eps': eps,
			 'momentum': Tensor.zeros(Engine.parameters.size),
			 'squared_grad': Tensor.zeros(Engine.parameters.size),
			 'step_count': Tensor.zeros(1)},
			update={
				'step_count': lambda self: self.step_count + 1,
				'momentum': lambda self: self.beta1*self.momentum + (1-self.beta1)*Engine.parameters.grad,
				'squared_grad': lambda self: self.beta2*self.squared_grad + (1-self.beta2)*Engine.parameters.grad**2
			},
			step=lambda self: self.lr * (self.momentum/(1-self.beta1**self.step_count)) /
							  ((self.squared_grad/(1-self.beta2**self.step_count))**0.5 + self.eps)
		)
