# Multi-Actor-Attention-Critic
Code for [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1810.02912) (Iqbal and Sha, ICML 2019)

## Requirements
* Python 3.6.1 (Minimum)
* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

The versions are just what I used and not necessarily strict requirements.

## How to Run

```shell
git clone https://github.com/yuchen-x/MAAC.git
cd MAAC
git checkout yuchen_pomdp_a_h_c_obs
cd ..
pip install tensorflow
pip install tensorboardx
git clone https://github.com/openai/baselines.git
cd baselines 
git checkout 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
pip install -e .
cd ..
pip install gym==0.9.4
git clone https://github.ccs.neu.edu/ycx424/multiagent-envs.git
cd multiagent-envs
git checkout yuchen
pip install -e .
cd ..
git clone https://github.com/shariqiqbal2810/multiagent-particle-envs.git
cd multiagent-particle-envs
vim multiagent-particle-envs/multiagent/__init__.py
comment everything
pip install -e .
cd ..
cd MAAC
python main.py --env_id=simple_collect_treasure --buffer_length=1000




```
