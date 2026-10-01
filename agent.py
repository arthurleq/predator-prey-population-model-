import numpy as np


def wrap_angle(angle):
    """Wrap an angle (or an array of angles, in radians) to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Group:
    """
    Entities of one kind (preys, predators, carcasses).

    Each attribute is a numpy array with one row per entity, so that the whole
    group can be updated at once instead of entity by entity.
    """

    def __len__(self):
        return len(self.pos)

    @property
    def x(self):
        return self.pos[:, 0]

    @property
    def y(self):
        return self.pos[:, 1]

    def keep(self, mask):
        """Keep only the entities for which mask is True (used to remove dead agents)."""
        for name, values in list(vars(self).items()):
            setattr(self, name, values[mask])


class Population(Group):
    """State of all the agents of one species (preys or predators)."""

    def __init__(self):
        # position (x, y) of each agent
        self.pos = np.empty((0, 2))

        # direction (in radians), always kept in [-pi, pi)
        self.direction = np.empty(0)

        # energy will go down during the simulation, and if it reaches 0, the agent dies
        self.energy = np.empty(0)

        # countdown is used to pause the agent when it is giving birth
        self.countdown = np.empty(0, dtype=int)

    def add(self, pos, direction, energy):
        """
        Add agents to the population.

        pos: array (n, 2) of positions
        direction: array (n,) of directions (in radians)
        energy: array (n,) of energies
        """
        self.pos = np.concatenate([self.pos, pos])
        self.direction = np.concatenate([self.direction, wrap_angle(direction)])
        self.energy = np.concatenate([self.energy, energy])
        self.countdown = np.concatenate([self.countdown, np.zeros(len(pos), dtype=int)])


class Carcasses(Group):
    """
    Carcasses of the preys killed by predators.

    The energy of the dead prey is a reserve that any predator on the carcass can eat,
    until the reserve is empty or the countdown reaches 0 (the carcass rots).
    """

    def __init__(self):
        # position (x, y) of each carcass
        self.pos = np.empty((0, 2))

        # energy left in the carcass
        self.energy = np.empty(0)

        # number of time steps before the carcass disappears
        self.countdown = np.empty(0, dtype=int)

    def add(self, pos, energy, countdown):
        """
        Add carcasses.

        pos: array (n, 2) of positions
        energy: array (n,) of energy reserves
        countdown: number of time steps before they disappear
        """
        self.pos = np.concatenate([self.pos, pos])
        self.energy = np.concatenate([self.energy, energy])
        self.countdown = np.concatenate([self.countdown, np.full(len(pos), countdown, dtype=int)])
