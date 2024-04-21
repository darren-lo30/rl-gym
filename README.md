# OpenAI Gym
A collection of different reinforcement learning algorithms applied to OpenAI Gym environments. Only some environments are tested and supported


## Training
To train a model on an environment

```
python3 main.py --env ... --model ... train --config ...
```
where `env` is the gym env name, `model` is the model name and `config` are the hyperparameter configurations

For example, to run reinforce with baseline on CartPole and save the model to a file
```
python3 main.py --env "CartPole-v1" --model "reinforce_baseline" train --config ... --save_file ...
```

## Loading
To load a model from a file and run it on an environment
```
python3 main.py --env ... --model ... load --save_file ...
```