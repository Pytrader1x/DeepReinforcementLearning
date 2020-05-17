# Deep Reinforcement Learning Robot: Navigation
Deep Reinforcement Learning nano degree

Objective:

Train a bot using Q learning to collect as many yellow bananas while avoiding blue bananas.

![](robot_banana.gif)
# Environment description:

The environment is based on Unity ML-agents

The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.



Getting started

Installation requirements

You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the Udacity repository

Of course you have to clone this project and have it accessible in your Python environment

Then you have to install the Unity environment as described in the Getting Started section (The Unity ML-agant environment is already configured by Udacity)

Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the environment.

Finally, unzip the environment archive in the 'project's environment' directory and eventually adjust thr path to the UnityEnvironment in the code.

# Train a agent

Execute the provided notebook within this Nanodegree Udacity Online Workspace for "project #1 Navigation" (or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code )

Note :

Manually playing with the environment has not been implemented as it is not available with Udacity Online Worspace (No Virtual Screen)
Watching the trained agent playing in the environment has not been implemented neither, as it is not available with Udacity Online Worspace (No Virtual Screen) and not compatible with my personal setup (see Misc : Configuration used section)
Misc : Configuration used

This agent has been trained on the Udacity provided online workspace. This environment allows to use a Nvidia K80 GPU that is used for the training. (The headless / no visualization version of the Unity environment was thus used)

My setup is a "Deep Learning Dev Box", and is basically a Linux GPU Server, running Docker containers (using Nvidia Docker 2), serving Jupyter Lab notebooks which are accessed remotely via a web interface (or a ssh connection) : unfortunately this setup does not seem suitable to run Unity ML agent, with the GPU and providing a display for for the agent (See Unity documentation for more details)
