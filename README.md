# Flappy_Bird_RL
The Flappy Bird game is controlled using tabular RL algorithms.

## Table of Contents
- [Description of game](#description-of-game)
- [Description of Environment](#description-of-environment)
- [Requirements](#requirements)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)

## Description of game
Flappy Bird is an arcade-style game in which the player controls the bird Faby, which moves persistently to the right. The player is tasked with navigating Faby through pairs of pipes that have equally sized gaps placed at random heights. Faby automatically descends and only ascends when the player taps the touchscreen. Each successful pass through a pair of pipes awards the player one point. Colliding with a pipe or the ground ends the gameplay.  
(source: [Wikipedia](https://en.wikipedia.org/wiki/Flappy_Bird))

## Description of Environment
The [Flappy Bird environment](https://github.com/Talendar/flappy-bird-gym) is very similar to the game where the bird has constant downward acceleration and angular velocity till it is flapped. The pipes are generated randomly and the environment mimics the game perfectly. For more details on its working, please visit [game logic](https://github.com/Talendar/flappy-bird-gym/blob/main/flappy_bird_gym/envs/game_logic.py).

|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/83444423-b654-44a6-9fdd-3795c7b3a5e3" width="288" height="512" /> |
|:--:|
|`FlappyBird-v0`|

## Requirements
You need to install the following libraries to use this environment.
- `pip install gymnasium` - Gymnasium
- `pip install pygame` - pygame
- `pip install numpy` - NumPy
- `pip install matplotlib` - Matplotlib
- `pip install flappy-bird-gym` - flappy-bird-gym

## State Space
The observation space only includes horizontal and vertical distances from the centre of pipe's gap. Since the bird's orientation and velocity are critical for optimal action, observations from the preceding four time-steps are consolidated into a cohesive state space.
$$state = (Hz_{t-4},Vr_{t-4},Hz_{t-3},Vr_{t-3},Hz_{t-2},Vr_{t-2},Hz_{t-1},Vr_{t-1})$$ 

The observation range is rescaled to one-fourth horizontally and one-third vertically for the implementation of tabular methods.

## Action Space
The agent can take only 2 actions:
- 0 - No Flap
- 1 - Flap

## Reward Function
The objective of the bird is to maximize pipe crossings while avoiding collisions with a pipe or the ground.
- When the bird crosses a pipe: +10
- When the bird collides with a pipe or ground: -10

## Algorithms Implemented
Tabular Model-Free algorithms are tested on this environment like:
- Q-learning
- SARSA-Lambda
- SARSA-Backwards

## Results
- Training

|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/c7d34c42-07f3-4caa-8a1f-cf4894d388c8" width="550" height="450"/>| 
|:--:|
|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/ddb0af62-19ce-47e7-92e2-89bd8f8acfb0" width="550" height="450" /> |
|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/e457caf2-d318-4df8-8403-d9c24f7babe0" width="550" height="450"/>|

- Testing

|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/26908144-6c43-485f-a048-c6ec5c8a08e6" width="550" height="450"/>| 
|:--:|
|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/1d9b5e56-254a-4390-8c79-009228e6f71f" width="550" height="450" /> |
|<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/7e37ae42-97cc-4b1d-879d-0f67686a40f6" width="550" height="450"/>|


For more information on this environment, please visit its [Documentation](https://github.com/Talendar/flappy-bird-gym).

<!-- |<img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/ddb0af62-19ce-47e7-92e2-89bd8f8acfb0" width="245" height="225" /> | <img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/c7d34c42-07f3-4caa-8a1f-cf4894d388c8" width="245" height="225"/>| <img src="https://github.com/RaviAgrawal-1824/Flappy_Bird_RL/assets/109269344/e457caf2-d318-4df8-8403-d9c24f7babe0" width="245" height="225"/>|
|:--:|:--:|:--:| -->
