import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import random
from sklearn.model_selection import train_test_split


from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
import keras
import warnings; warnings.simplefilter('ignore')
from datetime import datetime
import itertools 
import argparse
import re
import os 
import pickle 
import time

from sklearn.preprocessing import StandardScaler

def get_data():

	df = pd.read_csv('AMZN.csv')
	print(df.shape)
	return df.values #.values only returns the numpy array


def get_chart():

	data=get_data()
	train_data, test_data= train_test_split(data, test_size=0.5,random_state = 42,shuffle=False)
	print("for training data")

	plt.plot(train_data)
	plt.show()

	print("now testing data")

	plt.plot(test_data)
	plt.show()


def test_action_chart(action,string):
	data=get_data()
	train_data, test_data= train_test_split(data, test_size=0.5,random_state = 42,shuffle=False)
	flat_list=[]
	if string == "test":
		for sublist in test_data:
			for item in sublist:
				flat_list.append(item)
	elif string == "train":
		for sublist in train_data:
			for item in sublist:
				flat_list.append(item)
	
	c= []
	for i in range(0,len(action)):
		d=[flat_list[i],action[i]]
		c.append(d)
	print(c)
	df = pd.DataFrame(dict(flat_list=flat_list, action=action))
	#plt.plot(c)
	#plt.show()
	My_list = [*range(len(flat_list))] 
	fig, ax = plt.subplots(figsize=(20, 10))
	colors = {0:'red', 1:'blue', 2:'green'}
	if string=="train":
		plt.ylim(300, 900)
	else:
		plt.ylim(1500,2500)
	ax.plot(flat_list)
	ax.scatter(My_list,df['flat_list'], c=df['action'].apply(lambda x: colors[x]),s=80)
	plt.show()





def get_scaler(env):
	states=[]

	for _ in range(env.n_step):
		action = np.random.choice(env.action_space)
		state,reward,done,info = env.step(action)
		states.append(state)
		if done: 
			break
	scaler = StandardScaler()
	scaler.fit(states)
	return scaler

def maybe_make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)



	

class MultiStockEnv: 
	def __init__(self,data,initial_investment):
		self.stock_price_history = data
		self.n_step, self.n_stock = self.stock_price_history.shape
		

		self.initial_investment = initial_investment
		self.cur_step = None
		self.stock_owned = None
		self.stock_price = None
		self.cash_in_hand = None
		self.bought=None

		self.action_space = np.arange(3**self.n_stock)



		#0 1 2 sell hold buy


		self.action_list = list(map(list,itertools.product([0,1,2],repeat = self.n_stock)))
		# gives the list of all the actions that can be taken 

		self.state_dim = self.n_stock*2 + 1
		
		# dimension of state where our state is first the LTp of all stocks then the no of stocks we own and then the money we have in our account remaining
		self.reset()

	def reset(self):

		self.cur_step = 0 #current step first day
		self.stock_owned = np.zeros(self.n_stock) # 0
		self.stock_price = self.stock_price_history[self.cur_step] #current price of each stock
		self.cash_in_hand = self.initial_investment #2K
		return self._get_obs() #state function

	def step(self,action):
		assert action in self.action_space #action exiists in our action space

		prev_val = self._get_val(action) #current value to prev val

		self.cur_step += 1 #next day and update
		self.stock_price = self.stock_price_history[self.cur_step]

		#calling the trade
		if self.bought==True:
			s=1
		else:
			s=0
		self._trade(action)
		
		#getting the new value after trade

		cur_val = self._get_val(action)

		if s==1 and action==2:
			reward = 0
		elif s==0 and action==0:
			reward = 0
		else:
			reward = cur_val - prev_val



		


		done = self.cur_step == self.n_step - 1

		info = {'cur_val': cur_val}

		return self._get_obs(), reward, done , info 

	def _get_obs(self):
		#state and obs are same in this context
		obs = np.empty(self.state_dim)
		obs[:self.n_stock] = self.stock_owned #no of stock owned this should be size 3
		obs[self.n_stock:2*self.n_stock] = self.stock_price # stock prices this should also be size 3
		obs[-1] = self.cash_in_hand #size 1 since the capital we have is a single scalar value
		return obs
		#[3 values of no of stocks owned, 3 calues of current prices of stocks and one valueof cash in hand]

	def _get_val(self,action):
		#print(self.stock_owned.dot(self.stock_price) +self.cash_in_hand)
		#print(action)
		return self.stock_owned.dot(self.stock_price) +self.cash_in_hand

	def _trade(self, action):
		#here we do trade 
		action_vec=self.action_list[action]

		sell_index = []
		buy_index= []
		for i,a in enumerate(action_vec):
			if a == 0:
				sell_index.append(i)
			elif a== 2:
				buy_index.append(i)

		if sell_index:
			for i in sell_index:
				self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
				self.stock_owned[i]= 0
				self.bought=False
			
		

		# o sell 1 hold and 2 buy and we buy as much as we can and sell as much as we can as we get the order
		if buy_index:
			can_buy= True

			while can_buy:
				for i in buy_index:
					if self.cash_in_hand > self.stock_price[i]:
						self.stock_owned[i] += 1
						self.cash_in_hand -= self.stock_price[i]
						self.bought=True
					else:
						can_buy=False

def OurModel(input_shape, action_space):
	losses=[]
	X_input = Input(input_shape)
	X = Dense(512, input_shape=(input_shape,), activation="relu", kernel_initializer='he_uniform')(X_input)
	X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
	X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
	X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
	#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=0.01,
    #decay_steps=1000,
    #decay_rate=0.)
	#optimizer = keras.optimizers.Adam(learning_rate=0.000)
	model = Model(inputs = X_input, outputs = X)
	model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
	#model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
	#losses.append[mse]
	#model.summary()
	return model

class DQNAgent(object):
    def __init__(self,state_size,action_size):
       
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1 # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99 # change this to 0.999
        self.batch_size = 32
        self.train_start = 2000
        self.lost=[]

        self.model = OurModel(input_shape=self.state_size, action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
        	
        	if self.epsilon > self.epsilon_min:
        		self.epsilon *= self.epsilon_decay

    def act(self, state):
    	
    	if np.random.rand() <= self.epsilon:
    		return np.random.choice(self.action_size)
    	else:
    		act_values = self.model.predict(state)
    	
    	return np.argmax(act_values[0])

    def replay(self,state,action,reward,next_state,done):
    	
    	if len(self.memory) < self.train_start:
    		return


    	
    	# Randomly sample minibatch from the memory
    	

    	minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
    	state = np.zeros((self.batch_size, self.state_size))
    	next_state = np.zeros((self.batch_size, self.state_size))
    	action, reward, done = [], [], []


    	#state=[minibatch[i][0] for i in range(self.batch_size)]
    	#action=[minibatch[i][1] for i in range(self.batch_size)]
    	#reward=[minibatch[i][2] for i in range(self.batch_size)]
    	#state=[minibatch[i][3] for i in range(self.batch_size)]
    	#done=[minibatch[i][4] for i in range(self.batch_size)]


    	for i in range(self.batch_size):
    		state[i] = minibatch[i][0]
    		action.append(minibatch[i][1])
    		reward.append(minibatch[i][2])
    		next_state[i] = minibatch[i][3]
    		done.append(minibatch[i][4])
    
    	target = self.model.predict(state)
    	target_next = self.model.predict(next_state)

    	for i in range(self.batch_size):

    		if done[i]:
    			target[i][action[i]] = reward[i]
 
    		else:
    			target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))


    	history=self.model.fit(state, target, batch_size=self.batch_size, verbose=0,epochs=1)
    	self.lost.extend(history.history['accuracy'])



    	
 
    	



    def load(self, name):
    	self.model = keras.models.load_model(name)
    def save(self, name):
        self.model.save(name)


loss_model=[]
action_test=[]
def play_one_episode(agent, env, is_train,e):

	state = env.reset()
	action_test=[]
	state = scaler.transform([state])
	done = False
	

	while not done:
		action = agent.act(state)
		next_state, reward, done, info = env.step(action)
		next_state = scaler.transform([next_state])
		if is_train == 'train':
			
			history=agent.replay(state,action,reward,next_state,done)
			agent.remember(state,action,reward,next_state,done)
		
		action_test.append(action)

		state=next_state

			


		if done==1 and e>=2 and is_train=='train': 
			n = len(agent.lost)
			get_sum = sum(agent.lost)
			mean=get_sum/n 
			agent.lost=[]
			loss_model.append(mean)
			#print(agent.lost)

	action_test.append(action)







	return {'info': info['cur_val'], 'loss':loss_model, 'action_test':action_test}

if __name__ == '__main__':

	models_folder = 'linear_rl_trader_models'
	rewards_folder = 'linear_rl_trader_rewards'
	num_episodes = 50
	
	initial_investment = 2000

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', type = str ,required = True,
		help='either "train" or "test"')
	args = parser.parse_args()
	maybe_make_dir(models_folder)
	maybe_make_dir(rewards_folder)

	data = get_data()
	n_timesteps, n_stock = data.shape #n_stock or n_stocks

	train_data, test_data= train_test_split(data, test_size=0.5,random_state = 42,shuffle=False)

	env = MultiStockEnv(train_data, initial_investment)

	state_size = env.state_dim
	action_size = len(env.action_space)
	agent = DQNAgent(state_size, action_size)
	scaler = get_scaler(env)

	portfolio_value = []
	get_chart()






	if args.mode == 'test': 
		num_episodes = 1
		agent.load(f'{models_folder}')

		env = MultiStockEnv(test_data, initial_investment)

		agent_epsilon = 0.01

		for e in range (num_episodes):
			t0 = datetime.now()
			val = play_one_episode(agent,env,args.mode,e)
			dt = datetime.now() - t0
			print(f"episode: {e+1}/{num_episodes}, episode end value: {val['info']:.2f},duration:{dt}")
			portfolio_value.append(val['info'])


		test_action_chart(val['action_test'],"test")
		










	if args.mode == 'train':
		for e in range (num_episodes):
			t0 = datetime.now()
			val = play_one_episode(agent,env,args.mode,e)
			dt = datetime.now() - t0
			print(f"episode: {e+1}/{num_episodes}, episode end value: {val['info']:.2f},duration:{dt}")
			portfolio_value.append(val['info'])
			if (e+1) % 10==0:
				test_action_chart(val['action_test'],"train")
		print("loss is :")


		print(val['loss'])

		print("Now testing.")

		agent.save(f'{models_folder}')
			
		np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
		agent_epsilon = 0.01
		num_episodes = 3
		args.mode = 'test'
		env = MultiStockEnv(test_data, initial_investment)


		for e in range (num_episodes):
			t0 = datetime.now()
			val = play_one_episode(agent,env,args.mode,0)
			dt = datetime.now() - t0
			print(f"episode: {e+1}/{num_episodes}, episode end value: {val['info']:.2f},duration:{dt}")

		test_action_chart(val['action_test'],"test")

		print("The plot for accuracy is ")

		plt.plot(val['loss'])
		plt.show()



		






	
