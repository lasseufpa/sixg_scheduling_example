import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from agents.marl_test import MARLTest
from associations.simple import SimpleAssociation
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.simple import SimpleTraffic

seed = 10

marl_comm_env = MARLCommEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    SimpleAssociation,
    "simple",
    "agent_marl_test",
    seed,
    obs_space=MARLTest.get_obs_space,
    action_space=MARLTest.get_action_space,
    number_agents=2,
)
marl_test_agent = MARLTest(
    marl_comm_env,
    marl_comm_env.comm_env.max_number_ues,
    marl_comm_env.comm_env.max_number_basestations,
    marl_comm_env.comm_env.num_available_rbs,
)
marl_comm_env.comm_env.set_agent_functions(
    marl_test_agent.obs_space_format,
    marl_test_agent.action_format,
    marl_test_agent.calculate_reward,
)

register_env("marl_comm_env", lambda config: marl_comm_env)

config = (
    PPOConfig()
    .rollouts(num_rollout_workers=0, enable_connectors=False)
    .resources(num_gpus=0)
    .environment("marl_comm_env")
    .framework("torch")
)

algo = config.build()

total_train_steps = 1
for i in range(total_train_steps):
    result = algo.train()
    print(pretty_print(result))

total_test_steps = 10
obs, _ = marl_comm_env.reset(seed=seed)
for step in np.arange(total_test_steps):
    obs = marl_comm_env.observation_space.sample()
    sched_decision = algo.compute_actions(obs)
    obs, reward, terminated, truncated, info = marl_comm_env.step(
        sched_decision
    )
    if terminated["__all__"]:
        break
