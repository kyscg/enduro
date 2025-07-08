"""
file: 
    infer.py
description:
    loads model checkpoints and generates atari enduro gameplay video.
    also prints test average of any number of episodes required
url:
    https://kyscg.github.io/2025/07/11/dqnenduro
author:
    kyscg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio
import numpy as np
from PIL import Image
from collections import deque

import ale_py
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameter cache
MODEL_PATH = "path to checkpoint"
NUM_EVAL_EPISODES = 7 # change this to 100 or 200
VIDEO_FILENAME = "path to video"
VIDEO_FPS = 30
FRAME_SKIP = 4
STACK_SIZE = 4


def preprocess_frame(frame): # taken from train.py
    img = Image.fromarray(frame)
    img = img.convert('L')
    img = img.resize((84, 84), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)
    processed_frame = np.array(img, dtype=np.uint8)
    return processed_frame

class DQN(nn.Module): # taken from train.py
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._feature_size = self._get_conv_output_size((STACK_SIZE, 84, 84))
        self.fc1 = nn.Linear(self._feature_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        o = F.relu(self.conv1(dummy_input))
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

def generate_gameplay_video():

    env = gym.make('ALE/Enduro-v5', render_mode='rgb_array')
    num_actions = env.action_space.n
    print(f"Environment: ALE/Enduro-v5 | Action Space: {num_actions} actions.")

    policy_net = DQN(num_actions).to(device)
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        policy_net.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        env.close()
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    frames_to_record = [] # store all RGB frames
    total_rewards = []

    print(f"Generating gameplay video for {NUM_EVAL_EPISODES} episodes...")

    for episode in range(NUM_EVAL_EPISODES):
        state_raw, _ = env.reset()
        rendered_frame = env.render()
        if rendered_frame is not None:
            frames_to_record.append(rendered_frame)

        state_processed = preprocess_frame(state_raw)
        stacked_frames_deque = deque([state_processed] * STACK_SIZE, maxlen=STACK_SIZE)
        current_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(current_stacked_state, dtype=torch.float32).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax(1).item() # greedy policy 

            reward_sum_over_skip = 0
            frame_skip_done = False
            for _ in range(FRAME_SKIP):
                next_frame_raw, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rendered_frame = env.render()
                if rendered_frame is not None:
                    frames_to_record.append(rendered_frame)

                reward_sum_over_skip += reward
                if done:
                    frame_skip_done = True
                    break

            next_frame_processed = preprocess_frame(next_frame_raw)
            stacked_frames_deque.append(next_frame_processed)
            next_stacked_state = np.array(stacked_frames_deque, dtype=np.uint8)

            current_stacked_state = next_stacked_state
            episode_reward += reward_sum_over_skip

            if frame_skip_done:
                done = True

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{NUM_EVAL_EPISODES} finished with reward: {episode_reward:.2f}")

    env.close()

    if frames_to_record:
        print(f"Saving video to {VIDEO_FILENAME} with {len(frames_to_record)} frames...")
        imageio.mimwrite(VIDEO_FILENAME, frames_to_record, fps=VIDEO_FPS, codec='libx264', quality=9)
        print(f"Total rewards for {NUM_EVAL_EPISODES} episodes: {total_rewards}")
        print(f"Average reward per episode: {np.mean(total_rewards):.2f}")
    else:
        print("No frames were recorded. Video not generated.")

if __name__ == '__main__':
    generate_gameplay_video()
