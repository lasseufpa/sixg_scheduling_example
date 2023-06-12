from typing import Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from sixg_radio_mgmt import Agent, CommunicationEnv


class MARLTest(Agent):
    def __init__(
        self,
        env: Union[CommunicationEnv, AECEnv],
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        allocation_rbs = [
            np.zeros(
                (self.max_number_ues, self.num_available_rbs[basestation])
            )
            for basestation in np.arange(self.max_number_basestations)
        ]
        for basestation in np.arange(self.max_number_basestations):
            ue_idx = 0
            rb_idx = 0
            while rb_idx < self.num_available_rbs[basestation]:
                if obs_space[basestation][ue_idx] == 1:
                    allocation_rbs[basestation][ue_idx][rb_idx] += 1
                    rb_idx += 1
                ue_idx += 1 if ue_idx + 1 != self.max_number_ues else -ue_idx

        return np.array(allocation_rbs)

    def obs_space_format(self, obs_space: dict) -> np.ndarray:
        return np.zeros(4)

    def calculate_reward(self, obs_space: dict) -> float:
        return 0

    def action_format(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        return np.array(action)

    @staticmethod
    def get_action_space() -> dict:
        return {
            "player_0": spaces.Box(low=-1, high=1, shape=(2 * 2 * 2,)),
            "player_1": spaces.Box(low=-1, high=1, shape=(2 * 2 * 2,)),
        }

    @staticmethod
    def get_obs_space() -> dict:
        return {
            "player_0": spaces.Box(
                low=0, high=np.inf, shape=(2 * 2,), dtype=np.float64
            ),
            "player_1": spaces.Box(
                low=0, high=np.inf, shape=(2 * 2,), dtype=np.float64
            ),
        }
