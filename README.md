# Playing Enduro with Deep Reinforcement Learning

This repository contains an implementation of the Deep Q-Network (DQN) algorithm, as described in the seminal paper "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013/2015), applied to the Atari 2600 game Enduro using PyTorch and Gymnasium.

The project demonstrates the core components of DQN and includes key optimizations for stable and efficient training, such as `uint8` frame storage for memory efficiency and reward clipping for improved learning stability.

Read accompanying [blog post](https://kyscg.github.io/2025/07/11/dqnenduro) and watch [video](https://youtu.be/kRzhB5Fhd8M)

## Repository Structure

```
enduro/
├── logs/                     # Directory containing training log files for different variants
├── models/                   # Directory to store trained model checkpoints
│   └── dqn_enduro_best.pth   # The best performing trained model checkpoint
├── scripts/                  # Python scripts for training and evaluation
│   ├── train.py              # Script to train the DQN agent
│   └── infer.py              # Script to generate gameplay videos
├── videos/                   # Directory for generated gameplay videos
└── requirements.txt          # List of Python dependencies
```

## Features

- **Deep Q-Network (DQN) Architecture**: Implementation of the convolutional neural network architecture proposed by Mnih et al.
- **Experience Replay Buffer**: Stores and samples past transitions to break temporal correlations during training.
- **Memory Optimization**: Efficient `uint8` storage for states in the replay buffer, significantly reducing RAM usage.
- **Reward Clipping**: Rewards are clipped to −1,0,1 to maintain training stability and prevent large rewards from dominating the loss.


## Installation

1. Clone the repository

```bash
git clone https://github.com/kyscg/enduro.git
cd enduro
```

2. Create a virtual environment

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows (why?):
.\venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the agent

The `train.py` script trains the DQN agent on the Enduro environment. You can customize hyperparameters within the script. Training logs (e.g., episode rewards, epsilon values) will be printed to the console. You might want to redirect these to a file for persistent logging, which can then be moved to the `logs/` directory.

### Generating gameplay video

The `infer.py` script will generate gameplay video for as many episodes as required (change in the script) until termination. Make sure you point the paths to the checkpoints correctly. The best checkpoint has been provided in `models/` for reference.

## Results and Videos

There were a lot of variants trained, with different results. Two log files have been attached that show how agents collect rewards in this setting. Two things that improved performance significantly from test average of 100 to 320 to 481 was 1) using `uint8` for memory management in the replay buffer and 2) clipping rewards before stacking them into the buffer.

Overall, an average test performance of 481 was achieved which is comparable to the results in the paper.

## References

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
2. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

---
