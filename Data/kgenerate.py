#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:30:52 2020

@author: jtsatsaros
"""


from typing import List, Union
from scml.scml2020.utils import anac2020_config_generator, anac2020_world_generator, anac2020_assigner
from scml.scml2020 import SCML2020Agent
from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent, RandomAgent
from myagent import MyAgent
import multiprocessing


def generate_world(n_steps: int, n_processes: int, competitors: List[Union[str, SCML2020Agent]], n_agents_per_competitor):
    config = anac2020_config_generator(
        n_competitors=len(competitors),
        n_agents_per_competitor=n_agents_per_competitor,
        n_steps=n_steps,
        n_processes=n_processes
    )
    assigned = anac2020_assigner(
        config,
        max_n_worlds=None,
        n_agents_per_competitor=n_agents_per_competitor,
        competitors=competitors,
        params=[dict() for _ in competitors],
    )
    return [anac2020_world_generator(**(a[0])) for a in assigned]


def computeHelper(trial: int):
    COMPETITORS = [DecentralizingAgent, BuyCheapSellExpensiveAgent, RandomAgent, MyAgent]
    for step in [200]:
        for level in [4]:
            worlds = generate_world(step, level, COMPETITORS, 1)
            for i, world in enumerate(worlds):
                            world.run()
                            world.stats_df.to_csv(f'omerStats{i}_{step}_{level}_{trial}.csv')

def compute(n_trials):
      with multiprocessing.Pool() as pool:
        pool.map(computeHelper, n_trials)


if __name__ == "__main__":
    
    trials = [x for x in range(2000)]
    compute(trials)

    
    
   