# Enviroments in Offline RL

Implementations of Gridworld, Modelwin, and Modelfail to experiment with offline RL. The implementation is inspired by the implementation of NChainEnv found at
[https://github.com/openai/gym/blob/master/gym/envs/toy_text/nchain.py].

Please see [https://arxiv.org/pdf/1604.00923.pdf] for the description of these
environments.

- Grid World
    // Implemented in GridWorld.py.
    See [https://github.com/maximecb/gym-minigrid] for an existing implementation.

- ModelWin
    Implemented in ModelWin.py.
    step(): runs one step in ModelWin.
    reset(): be sure to call reset when done==True.

- ModelFail
    Implemented in ModelFail.py.
    step(): runs one step in ModelFail
    reset(): be sure to call reset when done==True.
