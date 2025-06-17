import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
from collections import deque

import gymnasium as gym
import ale_py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# hyperparameter cache
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_CAPACITY = 100000
LEARNING_RATE = 0.00025
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 250000
TARGET_UPDATE = 10000
NUM_EPISODES = 500
INITIAL_COLLECT_STEPS = 50000
FRAME_SKIP = 4
STACK_SIZE = 4

def preprocess_frame(frame):

    img = Image.fromarray(frame)
    img = img.convert('L')

    img = img.resize((84, 84), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)

    processed_frame = np.array(img, dtype=np.uint8)
    return processed_frame

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4) # 32 * 20 * 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 64 * 9 * 9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # 64 * 7 * 7

        self._feature_size = self._get_conv_output_size((STACK_SIZE, 84, 84))

        self.fc1 = nn.Linear(self._feature_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self, shape):
        dummy = torch.zeros(1, *shape)
        o = F.relu(self.conv1(dummy))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))

        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states_raw, actions, rewards, next_states_raw, dones = zip(*transitions)

        states = torch.tensor(np.array(states_raw), dtype=torch.float32).to(device) / 255.0
        next_states = torch.tensor(np.array(next_states_raw), dtype=torch.float32).to(device) / 255.0

        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, state_shape, action_dim, lr, gamma, eps_start, eps_end, eps_decay, target_update_freq, device):
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.device = device

        self.policy_net = DQN(action_dim).to(device)
        self.target_net = DQN(action_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, alpha=0.95, eps=0.01, weight_decay=0)
        self.criterion = nn.SmoothL1Loss()

        self.steps_done = 0


    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)

        if random.random() < eps_threshold:
            return torch.tensor([random.randrange(self.action_dim)], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0

                q_values = self.policy_net(state_tensor)

                return q_values.argmax(1).view(1, 1)

    def optimize_model(self, replay_buffer):

        if len(replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones.float()))

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f'Target network updated at step {self.steps_done}')

def train():

    env = gym.make('ALE/Enduro-v5')
    num_actions = env.action_space.n
    print(f'Environment: ALE/Enduro-v5 | Action Space: {num_actions} actions.')

    agent = DQNAgent(
        state_shape=(STACK_SIZE, 84, 84),
        action_dim=num_actions,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        target_update_freq=TARGET_UPDATE,
        device=device
    )

    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    total_frames = 0
    episode_rewards = []

    print("--- Starting Initial Experience Collection (Random Actions) ---")
    print(f"Collecting {INITIAL_COLLECT_STEPS} steps to fill replay buffer...")

    state_raw, _ = env.reset()
    state_processed = preprocess_frame(state_raw)

    stacked_frames_deque = deque([state_processed] * STACK_SIZE, maxlen=STACK_SIZE)
    current_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

    for i in range(INITIAL_COLLECT_STEPS):
        action = env.action_space.sample()

        next_frame_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_frame_preprocessed = preprocess_frame(next_frame_raw)
        stacked_frames_deque.append(next_frame_preprocessed)
        next_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

        clipped_reward_sum = np.sign(reward) if reward != 0 else 0
        replay_buffer.push(current_stacked_state, action, clipped_reward_sum, next_stacked_state, done)

        current_stacked_state = next_stacked_state
        total_frames += 1

        if done:
            state_raw, _ = env.reset()
            state_processed = preprocess_frame(state_raw)
            stacked_frames_deque = deque([state_processed] * STACK_SIZE, maxlen=STACK_SIZE)
            current_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

        if (i + 1) % 10000 == 0:
            print(f"Collected {i + 1}/{INITIAL_COLLECT_STEPS} frames for initial buffer.")

    print(f'Initial experience collection finished. Replay buffer size: {len(replay_buffer)}')
    print('--- Starting Main Training Loop ---')

    for episode in range(NUM_EPISODES):
        state_raw, _ = env.reset()
        state_processed = preprocess_frame(state_raw)
        stacked_frames_deque = deque([state_processed] * STACK_SIZE, maxlen=STACK_SIZE)
        current_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

        episode_reward = 0
        done = False

        while not done:
            action_tensor = agent.select_action(current_stacked_state)
            action = action_tensor.item()

            reward_sum_over_skip = 0
            frame_skip_done = False
            for _ in range(FRAME_SKIP):
                next_frame_raw, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                reward_sum_over_skip += reward
                if done:
                    frame_skip_done = True
                    break

            next_frame_processed = preprocess_frame(next_frame_raw)

            stacked_frames_deque.append(next_frame_processed)
            next_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

            clipped_reward_sum = np.sign(reward_sum_over_skip) if reward_sum_over_skip != 0 else 0
            replay_buffer.push(current_stacked_state, action, clipped_reward_sum, next_stacked_state, done)

            current_stacked_state = next_stacked_state

            episode_reward += reward_sum_over_skip

            total_frames += 1
            agent.steps_done = total_frames

            agent.optimize_model(replay_buffer)

            if frame_skip_done:
                done = True

        episode_rewards.append(episode_reward)

        avg_reward_last_100 = np.mean(episode_rewards[-100:])

        current_epsilon = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-1. * agent.steps_done / agent.eps_decay)

        print(f"Episode {episode + 1}/{NUM_EPISODES} | Total Frames: {total_frames} | "
              f"Epsilon: {current_epsilon:.4f} | Reward: {episode_reward:.2f} | "
              f"Avg 100-episode Reward: {avg_reward_last_100:.2f}")

        if (episode + 1) % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"dqn_enduro_episode_{episode + 1}.pth")
            print(f"Model checkpoint saved after episode {episode + 1}")

    env.close()
    print("\n--- Training Finished! ---")

    print("Final average reward over last 100 episodes:", np.mean(episode_rewards[-100:]))

    torch.save(agent.policy_net.state_dict(), "dqn_enduro_final.pth")
    print("Final model saved as dqn_enduro_final.pth")

if __name__ == '__main__':
    train()
