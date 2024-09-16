import pettingzoo.atari
import pettingzoo.butterfly
import pettingzoo.classic
import pettingzoo.mpe
import pettingzoo.sisl
from .environment import PettingZooWrapper


def PettingZooEnv(args):
    if args.env_name == 'Atari':
        if args.scenario_name == 'mario_bros':
            env = pettingzoo.atari.mario_bros_v3.parallel_env()
    elif args.env_name == 'Butterfly':
        if args.scenario_name == 'knights_archers_zombies':
            env = pettingzoo.butterfly.knights_archers_zombies_v10.parallel_env(vector_state=False)
    elif args.env_name == 'Classic':
        if args.scenario_name == 'rock_paper_scissors':
            env = pettingzoo.classic.rps_v2.parallel_env()
    elif args.env_name == 'MPE':
        if args.scenario_name == 'simple_spread':
            env = pettingzoo.mpe.simple_spread_v3.parallel_env()
    elif args.env_name == 'SISL':
        if args.scenario_name == 'multiwalker':
            env = pettingzoo.sisl.multiwalker_v9.parallel_env()

    return PettingZooWrapper(env)
