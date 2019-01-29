import numpy as np
import chainer
from chainer import Chain, Variable, cuda
import chainer.functions as F
import chainer.links as L
from chainer.cuda import cupy as cp
from matplotlib import pyplot as plt
import argparse
from copy import copy

batch_size = 500
n_pretrain = 15
n_maxtrain = 30

class NeuralNet(chainer.Chain):
	def __init__(self, n_hidden=1000, w1=None, b1=None, w2=None, w3=None):
		super().__init__()
		with self.init_scope():
			self.l1 = L.Convolution2D(3, 4, 5, initialW=w1, initial_bias=b1)
			self.l2 = L.Linear(3136, n_hidden, nobias=True, initialW=w2)
			self.l3 = L.Linear(n_hidden, 10, nobias=True, initialW=w3)
	
	def __call__(self, x, unchain=False):
		h1 = F.relu(self.l1(x))
		if unchain:
			h1.unchain()
		h2 = F.relu(self.l2(h1))
		return self.l3(h2)
	
	def prun(self, threshold=0.1):
		xp = self.xp
		w1 = self.l1.W.data
		b1 = self.l1.b.data
		w2 = self.l2.W.data
		w3 = self.l3.W.data
		beta = xp.sum(w2 ** 2, axis=1)
		cbf = xp.sort(beta)[threshold*beta.shape[0]]
		filter = cbf <= beta
		w2 = w2[filter]
		w3 = w3.T[filter].T
		n_hidden = int(xp.sum(filter))
		if type(w1) is cp.core.core.ndarray:
			w1 = w1.get()
			b1 = b1.get()
			w2 = w2.get()
			w3 = w3.get()
			return NeuralNet(n_hidden, w1, b1, w2, w3).to_gpu(0), n_hidden
		else:
			return NeuralNet(n_hidden, w1, b1, w2, w3), n_hidden

def init_random():
	np.random.seed(rand_seed)
	cp.random.seed(rand_seed)

def get_acc(model, xs, ts):
	bs = []
	xp = model.xp
	for i in range(len(xs)//batch_size):
		bs.append(model(xs[i * batch_size:(i + 1) * batch_size]).data)
	ys = xp.argmax(xp.vstack(bs), axis=1)
	num_cors = xp.sum(ys == ts)
	acc = num_cors / ts.shape[0]
	if type(acc) is cp.core.core.ndarray:
		return float(acc.get())
	return float(acc)

def stack_result(tag, plot_data):
	if tag not in result_dct.keys():
		result_dct[tag] = {}
	for c, (label, val) in enumerate(plot_data.items()):
		if label in result_dct[tag].keys():
			result_dct[tag][label].append(val)
		else:
			result_dct[tag][label] = [val]
	
def plot_figure():
	fig = plt.figure()
	for tag in result_dct.keys():
		ax1 = fig.add_subplot(1,1,1)
		ax2 = ax1.twinx()
		for c, (label, vval) in enumerate(result_dct[tag].items()):
			val = np.mean(vval, axis=0)
			if label.startswith('acc'):
				print(f'{label} max:',np.max(val),'last:',val[-1])
				ax1.plot(np.arange(len(val))+1, val, color=f'C{c}', label=label, linewidth=1.)
			else:
				ax2.plot(np.arange(len(val))+1, val, color=f'C{c}', label=label, linewidth=1.)
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2)
		ax1.set_xlabel('epochs')
		ax1.set_ylabel('acc')
		ax1.grid(True)
		ax2.set_ylabel('cross entropy')
		ax2.set_ylim(ymax=5)
		plt.savefig(tag)
		plt.clf()
	
def main():
	init_random()
	model = NeuralNet()
	if args.gpu:
		cuda.get_device(0).use()
		model.to_gpu(0)

	train, test = chainer.datasets.get_cifar10()
	xs, ts = train._datasets
	txs, tts = test._datasets
	xs = cp.array(xs.reshape((len(xs),3,32,32)))
	ts = cp.array(ts)
	txs = cp.array(txs.reshape((len(txs),3,32,32)))
	tts = cp.array(tts)
	
	def train_model(num_epochs, model, unchain=False):
		init_random()
		xp = model.xp
		optimizer = chainer.optimizers.Adam()
		optimizer.setup(model)
		acc_train = []
		loss_train = []
		acc_test = []
		for i in range(num_epochs):
			loss_train_log = []
			n_loop = int(len(xs) // batch_size)
			optimizer.new_epoch()
			for j in range(n_loop):
				model.zerograds()
				x = xs[j * batch_size:(j + 1) * batch_size]
				t = ts[j * batch_size:(j + 1) * batch_size]
				t = Variable(t)
				y = model(x, unchain=unchain)
				loss = F.softmax_cross_entropy(y, t)
				loss.backward()
				optimizer.update()
				loss_train_log.append(xp.mean(loss.data))
			acc_train.append(get_acc(model, xs, ts))
			loss_train.append(float(np.sum(loss_train_log)/n_loop))
			acc_test.append(get_acc(model, txs, tts))
		return {'acc(train)':acc_train, 'acc(test)':acc_test, 'loss(train)':loss_train}
		
	if not(args.basic or args.pruning or args.iteration or args.minimal):
		data = train_model(n_maxtrain, model)
		stack_result('test.png', data)
	elif args.basic or args.pruning or args.iteration:
		pretrain_data = train_model(n_pretrain, model)
	
	if args.basic:
		# basic score
		model0 = model.copy('copy')
		base_data = train_model(n_maxtrain-n_pretrain, model0, args.unchain)
		base_train = pretrain_data['acc(train)']+base_data['acc(train)']
		base_test = pretrain_data['acc(test)']+base_data['acc(test)']
		base_loss = pretrain_data['loss(train)']+base_data['loss(train)']
		stack_result('basic_unchain.png' if args.unchain else 'basic.png',
					{'acc(train)':base_train,'acc(test)':base_test,'loss(train)':base_loss})
		del model0
	
	if args.pruning:
		# train pruned net
		model1, nodes1 = model.prun(0.65)
		model2, nodes2 = model.prun(0.75)
		prun_data1 = train_model(n_maxtrain-n_pretrain, model1, args.unchain)
		prun_data2 = train_model(n_maxtrain-n_pretrain, model2, args.unchain)
		prun_train1 = pretrain_data['acc(train)']+prun_data1['acc(train)']
		prun_test1 = pretrain_data['acc(test)']+prun_data1['acc(test)']
		prun_loss1 = pretrain_data['loss(train)']+prun_data1['loss(train)']
		prun_train2 = pretrain_data['acc(train)']+prun_data2['acc(train)']
		prun_test2 = pretrain_data['acc(test)']+prun_data2['acc(test)']
		prun_loss2 = pretrain_data['loss(train)']+prun_data2['loss(train)']
		stack_result('pruning_unchain.png' if args.unchain else 'pruning.png',
					{	f'acc(train1_{nodes1})':prun_train1,f'acc(test1_{nodes1}':prun_test1,
						f'acc(train2_{nodes2})':prun_train2,f'acc(test2_{nodes2}':prun_test2,
						f'loss(train1_{nodes1})':prun_loss1,f'loss(train2_{nodes2}':prun_loss2})
		del model1, model2

	if args.iteration:
		# train pruned net
		prun_train1 = pretrain_data['acc(train)'][:]
		prun_test1 = pretrain_data['acc(test)'][:]
		prun_loss1 = pretrain_data['loss(train)'][:]
		prun_train2 = pretrain_data['acc(train)'][:]
		prun_test2 = pretrain_data['acc(test)'][:]
		prun_loss2 = pretrain_data['loss(train)'][:]
		model1 = model.copy('copy')
		model2 = model.copy('copy')
		for i in range(n_maxtrain-n_pretrain):
			model1, nodes1 = model1.prun(0.067595123)  # (0.65^15)^f-1
			model2, nodes2 = model2.prun(0.142304101)  # (0.9^15)^f-1
			prun_data1 = train_model(1, model1, args.unchain)
			prun_data2 = train_model(1, model2, args.unchain)
			prun_train1 = prun_train1+prun_data1['acc(train)']
			prun_test1 = prun_test1+prun_data1['acc(test)']
			prun_loss1 = prun_loss1+prun_data1['loss(train)']
			prun_train2 = prun_train2+prun_data2['acc(train)']
			prun_test2 = prun_test2+prun_data2['acc(test)']
			prun_loss2 = prun_loss2+prun_data2['loss(train)']
		stack_result('iteration_unchain.png' if args.unchain else 'iteration.png',
					{	f'acc(train1_{nodes1})':prun_train1,f'acc(test1_{nodes1}':prun_test1,
						f'acc(train2_{nodes2})':prun_train2,f'acc(test2_{nodes2}':prun_test2,
						f'loss(train1_{nodes1})':prun_loss1,f'loss(train2_{nodes1}':prun_loss2})
		del model1, model2
	
	if args.minimal:
		# basic score
		model0 = NeuralNet(100)
		model0.to_gpu(0)
		pretrain_data = train_model(n_pretrain, model0)
		mini_data = train_model(n_maxtrain-n_pretrain, model0)
		mini_train = pretrain_data['acc(train)']+mini_data['acc(train)']
		mini_test = pretrain_data['acc(test)']+mini_data['acc(test)']
		mini_loss = pretrain_data['loss(train)']+mini_data['loss(train)']
		stack_result('minimal.png',
					{'acc(train_min)':mini_train,'acc(test_min)':mini_test,'loss(train_min)':mini_loss})
		del model0

if __name__ == '__main__':
	ps = argparse.ArgumentParser( description='ML Test' )
	ps.add_argument( '--gpu', '-g', action='store_true', help='Use GPU' )
	ps.add_argument( '--basic', '-b', action='store_true', help='Train Basic Neuralnet' )
	ps.add_argument( '--pruning', '-p', action='store_true', help='Train Pruning Neuralnet' )
	ps.add_argument( '--iteration', '-i', action='store_true', help='Iterate Pruning Neuralnet' )
	ps.add_argument( '--unchain', '-u', action='store_true', help='Unchain to Convolusion' )
	ps.add_argument( '--minimal', '-m', action='store_true', help='Minimal hidden size' )
	args = ps.parse_args()
	result_dct = {}
	rand_seed = 1
	main()
	rand_seed = 2
	main()
	rand_seed = 3
	main()
	plot_figure()
	
