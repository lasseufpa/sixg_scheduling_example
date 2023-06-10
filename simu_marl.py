import numpy as np
from tqdm import tqdm

from agents.marl_test import MARLTest
from associations.simple import SimpleAssociation
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.simple import SimpleTraffic

from pettingzoo.test import api_test

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

api_test(marl_comm_env, num_cycles=1000, verbose_progress=False)
exit()
round_robin = RoundRobin(
    comm_env,
    comm_env.max_number_ues,
    comm_env.max_number_basestations,
    comm_env.num_available_rbs,
)
comm_env.set_agent_functions(
    round_robin.obs_space_format,
    round_robin.action_format,
    round_robin.calculate_reward,
)

obs = comm_env.reset(seed=seed)[0]  # type: ignore
number_steps = 10
for step_number in tqdm(np.arange(comm_env.max_number_steps)):
    sched_decision = round_robin.step(obs)
    obs, _, end_ep, _, _ = comm_env.step(sched_decision)  # type: ignore
    if end_ep:
        comm_env.reset()
