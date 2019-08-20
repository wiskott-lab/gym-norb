from gym.envs.registration import register
from norb.norb import *

register(
    id='Norb-v0',
    entry_point='norb.norb:NorbEnv',
    kwargs={'scenario': NorbEnv.scenarios[0]}
)

register(
    id='Norb-v1',
    entry_point='norb.norb:NorbEnv',
    kwargs={'scenario': NorbEnv.scenarios[1]}
)

register(
    id='Norb-v2',
    entry_point='norb.norb:NorbEnv',
    kwargs={'scenario': NorbEnv.scenarios[2]}
)
