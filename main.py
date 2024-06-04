import gymnasium
import numpy
import torch

from torch.nn import ELU
from torch.nn import Module
from torch.nn import MSELoss

from torch.optim import Adam

from torch_optimizer import Lookahead
from torchrl.modules import NoisyLinear


class CombinedExperienceReplay(object):
	def __init__(self):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.memory_size = 1000000
		self.state_shape = 8
		self.action_shape = 1
		self.reward_shape = 1
		self.terminal_shape = 1
		self.batch_size = 512
		self.memory_counter = 0

		self.state_memory = numpy.zeros((self.memory_size, self.state_shape), dtype=numpy.float32)
		self.action_memory = numpy.zeros((self.memory_size, self.action_shape), dtype=numpy.int64)
		self.reward_memory = numpy.zeros((self.memory_size, self.reward_shape), dtype=numpy.float32)
		self.next_state_memory = numpy.zeros((self.memory_size, self.state_shape), dtype=numpy.float32)
		self.terminated_memory = numpy.zeros((self.memory_size, self.terminal_shape), dtype=bool)

	def save_to_memory(self, state, action, reward, next_state, terminated):
		index = self.memory_counter % self.memory_size

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.terminated_memory[index] = terminated

		self.memory_counter += 1

	def sample_from_memory(self):
		index = self.memory_counter % self.memory_size
		offset = index - 1
		batch = numpy.random.choice(offset, self.batch_size-1, replace=False)
		batch = numpy.append(batch, index)

		states = torch.tensor(self.state_memory[batch], dtype=torch.float32).to(self.device)
		actions = torch.tensor(self.action_memory[batch], dtype=torch.int64).to(self.device)
		rewards = torch.tensor(self.reward_memory[batch], dtype=torch.float32).to(self.device)
		next_states = torch.tensor(self.next_state_memory[batch], dtype=torch.float32).to(self.device)
		terminations = torch.tensor(self.terminated_memory[batch], dtype=torch.int64).to(self.device)

		return states, actions, rewards, next_states, terminations

	def is_sufficient(self):
		return (self.memory_counter % self.memory_size) > self.batch_size

class DuelingDeepQNetwork_MLP(Module):
	def __init__(self):
		super(DuelingDeepQNetwork_MLP, self).__init__()
		self.fc1 = NoisyLinear(8, 256, std_init=0.2)
		self.fc2 = NoisyLinear(256, 256, std_init=0.2)
		
		self.V_fc1 = NoisyLinear(256, 256, std_init=0.2)
		self.V = NoisyLinear(256, 1, std_init=0.2)
		
		self.A_fc1 = NoisyLinear(256, 256, std_init=0.2)
		self.A = NoisyLinear(256, 4, std_init=0.2)

		self.activation = ELU()

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

		self.loss = MSELoss().to(self.device)
		self.base_optimizer = Adam(self.parameters(), amsgrad=True)
		self.optimizer = Lookahead(self.base_optimizer)

	def forward(self, state):
		fc1 = self.activation(self.fc1(state))
		fc2 = self.activation(self.fc2(fc1))

		V_fc1 = self.activation(self.V_fc1(fc2))
		V = self.V(V_fc1)

		A_fc1 = self.activation(self.A_fc1(fc2))
		A = self.A(A_fc1)

		advAverage = torch.mean(A, dim=1, keepdim=True)

		Q = torch.add(V, (A - advAverage))

		return Q

	def save_networks(self):
		print("!.SAVING NETWORKS.!")
		torch.save(self.state_dict(), "DuelingDeepQNetwork_MLP.para")

	def load_networks(self):
		torch.load_state_dict(torch.load("DuelingDeepQNetwork_MLP.para"))

	def choose_action(self, state):
		state = torch.tensor(numpy.array([state]), dtype=torch.float32).to(self.device)
		with torch.no_grad():
			Q = self.forward(state)
		action = torch.argmax(Q).item()

		return action

class Agent(object):
	def __init__(self, online_network, target_network, memory):
		self.online_network = online_network
		self.target_network = target_network
		self.memory = memory
		self.polyak = 0.95
		self.gamma = 0.98

	def update_network_parameters(self, source, target):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

	def learn(self):

		if self.memory.is_sufficient():

			states, actions, rewards, next_states, terminations = self.memory.sample_from_memory()

			online_q_values = self.online_network(states).gather(1, actions)
			online_next_q_values = self.online_network(next_states)
			online_next_action = torch.argmax(online_next_q_values, dim=1, keepdim=True)
			target_q_next = self.target_network(next_states).gather(1, online_next_action)
			target_q_values = rewards + self.gamma * target_q_next * (1 - terminations)

			loss = self.online_network.loss(online_q_values, target_q_values)
			self.online_network.optimizer.zero_grad()
			loss.backward()
			self.online_network.optimizer.step()

			self.update_network_parameters(self.online_network, self.target_network)


if __name__ == "__main__":

	def train():

		env = gymnasium.make("LunarLander-v2", render_mode="human", continuous=False)

		memory = CombinedExperienceReplay()

		online_network = DuelingDeepQNetwork_MLP()

		target_network = DuelingDeepQNetwork_MLP()

		target_network.load_state_dict(online_network.state_dict())

		agent = Agent(online_network, target_network, memory)

		t = 0

		for e in range(1000):

			state = env.reset()[0]

			score = 0
			
			while True:

				action = online_network.choose_action(state)

				next_state, reward, terminated, truncated, info = env.step(action)

				done = terminated or truncated

				memory.save_to_memory(state, action, reward, next_state, done)

				agent.learn()

				state = next_state

				t+=1

				score += reward

				if done:
					break

			print(e, t, terminated, truncated, done, score)

		env.close()

	def test():
		pass


	train()
