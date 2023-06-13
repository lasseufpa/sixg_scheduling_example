import numpy as np
from pettingzoo.test import api_test, seed_test
from tqdm import tqdm

from agents.marr_test import MARRTest
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
    obs_space=MARRTest.get_obs_space,
    action_space=MARRTest.get_action_space,
    number_agents=2,
)
marl_test_agent = MARRTest(
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

marl_comm_env.reset(seed=seed)
for agent in marl_comm_env.agent_iter():
    obs, reward, termination, truncation, info = marl_comm_env.last()
    if termination:
        break
    sched_decision = marl_test_agent.step(agent, obs)
    marl_comm_env.step(sched_decision)
