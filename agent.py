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
            if isinstance(values, np.ndarray):
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

        # number of times each agent has reproduced
        self.reproductions = np.empty(0, dtype=int)

        # unique identifier of each agent, to follow it over time (the agents stay sorted by id)
        self.ids = np.empty(0, dtype=int)
        self.next_id = 0

        # (predators only) id of the carcass the agent is eating or ate last (-1 if none),
        # and the energy it took from it: a predator can only take a share of each carcass
        self.meal_carcass = np.empty(0, dtype=int)
        self.meal_eaten = np.empty(0)

    def add(self, pos, direction, energy):
        """
        Add agents to the population.

        pos: array (n, 2) of positions
        direction: array (n,) of directions (in radians)
        energy: array (n,) of energies
        """
        n = len(pos)
        self.pos = np.concatenate([self.pos, pos])
        self.direction = np.concatenate([self.direction, wrap_angle(direction)])
        self.energy = np.concatenate([self.energy, energy])
        self.countdown = np.concatenate([self.countdown, np.zeros(n, dtype=int)])
        self.reproductions = np.concatenate([self.reproductions, np.zeros(n, dtype=int)])
        self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + n)])
        self.next_id += n
        self.meal_carcass = np.concatenate([self.meal_carcass, np.full(n, -1)])
        self.meal_eaten = np.concatenate([self.meal_eaten, np.zeros(n)])


class Carcasses(Group):
    """
    Carcasses of the preys killed by predators.

    The energy of the dead prey is a reserve that any predator on the carcass can eat,
    until the reserve is empty or the countdown reaches 0 (the carcass rots).
    """

    def __init__(self):
        # position (x, y) of each carcass
        self.pos = np.empty((0, 2))

        # energy left in the carcass, and energy of the prey when it died
        self.energy = np.empty(0)
        self.initial_energy = np.empty(0)

        # number of time steps before the carcass disappears
        self.countdown = np.empty(0, dtype=int)

        # unique identifier of each carcass
        self.ids = np.empty(0, dtype=int)
        self.next_id = 0

    def add(self, pos, energy, countdown):
        """
        Add carcasses.

        pos: array (n, 2) of positions
        energy: array (n,) of energy reserves
        countdown: number of time steps before they disappear
        """
        n = len(pos)
        self.pos = np.concatenate([self.pos, pos])
        self.energy = np.concatenate([self.energy, energy])
        self.initial_energy = np.concatenate([self.initial_energy, energy])
        self.countdown = np.concatenate([self.countdown, np.full(n, countdown, dtype=int)])
        self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + n)])
        self.next_id += n
