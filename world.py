import numpy as np
from agent import Population, Carcasses, wrap_angle
from policies import prey_heuristic, predator_heuristic

class World:

    def __init__(self, width, height):

        # dimensions of the world
        self.width = width
        self.height = height

        # agents in the world (preys and predators), stored as arrays (see agent.py)
        self.preys = Population()
        self.predators = Population()

        # carcasses of the preys killed by predators
        self.carcasses = Carcasses()

        # decision rules of the agents (see policies.py)
        # they can be replaced by any function with the same signature (e.g. a neural network)
        self.prey_policy = prey_heuristic
        self.predator_policy = predator_heuristic

        # time step of the simulation
        self.time = 0

        # distance at which the agents can reproduce
        self.distance_reproduction = 1.0

        # distance at which the prey will run away from predators instead of looking for a mate
        self.distance_danger = 5.0

        # distance at which a predator can kill a prey, or eat a carcass
        self.distance_eating = 1.0

        # perception radius: an agent doesn't perceive the agents and carcasses further away
        self.perception_prey = 10.0
        self.perception_predator = 20.0

        # carrying capacity of the preys (logistic growth): a newborn prey survives
        # with a probability 1 - (number of preys) / carrying_capacity_prey
        self.carrying_capacity_prey = 200

        # limit for the number of predators in the world
        self.max_predators = 100

        # speed of the agents in the world
        self.speed_prey = 1.0
        self.speed_predator = 1.5

        # After each update, an agent can change its direction by a maximum of max_angular_change
        self.max_angular_change = np.pi / 8

        # max energy of each agent
        self.max_energy_prey = 150
        self.max_energy_predator = 150

        # energy needed for reproduction
        self.reproduction_energy_needed_prey = 150
        self.reproduction_energy_needed_predator = 150

        # energy cost for reproduction
        self.reproduction_energy_cost_prey = 30
        self.reproduction_energy_cost_predator = 30

        # when a predator kills a prey, the prey becomes a carcass and its energy a reserve
        # that every predator on the carcass can eat (they stay immobile while eating)
        # number of time steps before a carcass disappears
        self.carcass_duration = 30
        # energy taken from a carcass by each predator at each time step
        self.carcass_eating_rate = 10

        # energy gain by a prey at each time step
        self.energy_gain_prey = 1

        # energy lost by a predator at each time step
        self.energy_loss_predator = 1

        self.color_prey = "blue"
        self.color_predator = "orange"

    @property
    def mean_prey_energy(self):
        if len(self.preys) == 0:
            return 0
        return float(np.mean(self.preys.energy))

    @property
    def mean_predator_energy(self):
        if len(self.predators) == 0:
            return 0
        return float(np.mean(self.predators.energy))


    def wrap_vector(self, d):
        """Shortest equivalent of the vector(s) d (last axis = (dx, dy)) in the toric world."""
        size = np.array([self.width, self.height])
        return (d + size / 2) % size - size / 2

    def displacement(self, pos_from, pos_to):
        """
        Shortest vector from each position of pos_from to each position of pos_to in the toric world.

        pos_from: array (n, 2), pos_to: array (m, 2)
        return: dx, dy, two arrays (n, m)
        """
        d = self.wrap_vector(pos_to[None, :, :] - pos_from[:, None, :])
        return d[..., 0], d[..., 1]

    def spawn(self, population, positions):
        """Create agents at the given positions (array (n, 2)), with a random direction and energy."""
        n = len(positions)
        population.add(
            positions,
            direction=np.random.uniform(-np.pi, np.pi, n),
            energy=np.random.uniform(1, 150, n),
        )

    def add_agents(self, population, number=None, position=None):
        # if an int is given, we create that many agents with random positions
        # (no random agent by default when a position is given)
        if number is None:
            number = 0 if position is not None else 1
        positions = np.random.rand(number, 2) * [self.width, self.height]
        # if a tuple is given, we create an agent at that position
        if position is not None:
            positions = np.vstack([positions, position])
        self.spawn(population, positions)

    # method to add preys to the world
    def add_prey(self, number=None, position=None):
        """
        Add preys to the world. The preys can be added in two ways:
        number (int) : By specifying the number of preys to add. The preys will be added at random positions in the world.
        position (tuple) : By specifying the position of a single prey
        """
        self.add_agents(self.preys, number, position)

    # method to add predators to the world
    def add_predator(self, number=None, position=None):
        """
        Add predators to the world. The predators can be added in two ways:
        1. By specifying the number of predators to add. The predators will be added at random positions in the world.
        2. By specifying the position of a single predator
        """
        self.add_agents(self.predators, number, position)


    def remove_all_agents(self):
        """
        Remove all agents (preys and predators) from the world.
        """
        self.preys = Population()
        self.predators = Population()
        self.carcasses = Carcasses()

    def return_dict(self):
        """
        Return a snapshot of the world (copies of the arrays), used to draw the simulation afterwards:
        position, direction and energy of the preys and predators, which predators are eating a carcass,
        and the position and energy left of the carcasses.
        """
        return {
            "preys": {
                "pos": self.preys.pos.copy(),
                "direction": self.preys.direction.copy(),
                "energy": self.preys.energy.copy(),
            },
            "predators": {
                "pos": self.predators.pos.copy(),
                "direction": self.predators.direction.copy(),
                "energy": self.predators.energy.copy(),
                "eating": self.carcass_eaten() >= 0,
            },
            "carcasses": {
                "pos": self.carcasses.pos.copy(),
                "energy": self.carcasses.energy.copy(),
            },
        }

    def turn(self, population, actions):
        """
        Change the direction of the agents.

        actions: for each agent, proportion of max_angular_change in [-1, 1] (clipped if outside)
        """
        actions = np.clip(np.asarray(actions, dtype=float), -1, 1)
        population.direction = wrap_angle(population.direction + actions * self.max_angular_change)

    def move(self, population, speed, immobile=None):
        """
        Move the agents along their direction.

        immobile: optional mask of the agents that can't move during this step (e.g. predators eating a carcass)
        """
        # if the agent is in pause (giving birth), it doesn't move
        paused = population.countdown > 0
        population.countdown[paused] -= 1

        moving = ~paused
        if immobile is not None:
            moving &= ~immobile
        direction = population.direction[moving]
        velocity = speed * np.column_stack([np.cos(direction), np.sin(direction)])

        # toric world: if the agent goes out of the world, it appears on the other side
        population.pos[moving] = (population.pos[moving] + velocity) % [self.width, self.height]

    # Method to kill a prey if a predator is close enough
    def hunting(self, can_hunt):
        """
        Each predator kills the closest prey within reach; a prey can only be killed once.
        The dead prey becomes a carcass, and its energy a reserve for the predators.

        can_hunt: mask of the predators allowed to hunt (the ones eating a carcass don't hunt)
        """
        preys, predators = self.preys, self.predators
        if len(preys) == 0 or len(predators) == 0:
            return

        dx, dy = self.displacement(predators.pos, preys.pos)
        distance = np.hypot(dx, dy)
        # if the predator is close enough to the prey, it can kill it
        in_reach = distance < self.distance_eating

        # predators are handled one after the other, so that a prey can only be killed once
        alive = preys.energy > 0
        killed = []
        for i in np.flatnonzero(in_reach.any(axis=1) & can_hunt):
            # a prey already killed during this step can't be killed again
            candidates = np.flatnonzero(in_reach[i] & alive)
            if len(candidates) == 0:
                continue
            # a predator can only kill one prey at a time: the closest one
            prey = candidates[distance[i, candidates].argmin()]
            killed.append(prey)
            alive[prey] = False

        killed = np.array(killed, dtype=int)
        # the dead preys become carcasses: their energy is kept as a reserve
        self.carcasses.add(preys.pos[killed], preys.energy[killed], self.carcass_duration)
        # preys die
        preys.energy[killed] = 0

    def carcass_eaten(self):
        """
        For each predator, index of the carcass it is eating (the closest one within reach), or -1.
        A predator that is full doesn't eat, so it is free to leave the carcass.
        """
        n = len(self.predators)
        carcass = np.full(n, -1)
        if n == 0 or len(self.carcasses) == 0:
            return carcass

        dx, dy = self.displacement(self.predators.pos, self.carcasses.pos)
        distance = np.hypot(dx, dy)
        closest = distance.argmin(axis=1)
        eating = (
            (distance[np.arange(n), closest] < self.distance_eating)
            & (self.predators.energy < self.max_energy_predator)
        )
        carcass[eating] = closest[eating]
        return carcass

    def feeding(self):
        """
        The predators on a carcass take energy from its reserve (carcass_eating_rate each, at most).
        If the reserve is not enough for all of them, it is shared in proportion to what each one can eat.
        """
        carcass = self.carcass_eaten()
        eaters = np.flatnonzero(carcass >= 0)
        if len(eaters) == 0:
            return
        carcass = carcass[eaters]
        n_carcasses = len(self.carcasses)

        # what each predator can eat (it can't go over its max energy)
        wanted = np.minimum(self.carcass_eating_rate, self.max_energy_predator - self.predators.energy[eaters])
        # part of the demand that each carcass can satisfy
        demand = np.bincount(carcass, weights=wanted, minlength=n_carcasses)
        ratio = np.minimum(1, self.carcasses.energy / np.maximum(demand, 1e-12))
        eaten = wanted * ratio[carcass]

        self.predators.energy[eaters] += eaten
        self.carcasses.energy = np.maximum(
            self.carcasses.energy - np.bincount(carcass, weights=eaten, minlength=n_carcasses),
            0
        )

    def rotting(self):
        """The carcasses disappear when their countdown reaches 0 or when their reserve is empty."""
        self.carcasses.countdown -= 1
        self.carcasses.keep((self.carcasses.countdown > 0) & (self.carcasses.energy > 0))

    def mating(self, population, energy_needed, energy_cost, countdowns):
        """
        Reproduction of the agents of one population.

        Two agents reproduce if they are close enough and both have enough energy.
        Each agent reproduces at most once per time step, with the closest available partner.

        countdowns: pause of the two parents after the birth (female, male)
        return: positions of the newborns (array (n, 2))
        """
        newborns = np.empty((0, 2))

        # only the agents with enough energy can reproduce
        candidates = np.flatnonzero(population.energy >= energy_needed)
        if len(candidates) < 2:
            return newborns

        pos = population.pos[candidates]
        dx, dy = self.displacement(pos, pos)
        distance = np.hypot(dx, dy)
        np.fill_diagonal(distance, np.inf)
        close = distance < self.distance_reproduction

        available = np.ones(len(candidates), dtype=bool)
        females, males = [], []
        for a in np.flatnonzero(close.any(axis=1)):
            if not available[a]:
                continue
            partners = np.flatnonzero(close[a] & available)
            if len(partners) == 0:
                continue
            b = partners[distance[a, partners].argmin()]
            available[a] = available[b] = False
            females.append(candidates[a])
            males.append(candidates[b])

        if not females:
            return newborns
        females, males = np.array(females), np.array(males)

        # both parents lose some energy
        population.energy[females] = np.maximum(population.energy[females] - energy_cost, 0)
        population.energy[males] = np.maximum(population.energy[males] - energy_cost, 0)

        # countdown because they are exausted
        population.countdown[females] = countdowns[0]
        population.countdown[males] = countdowns[1]

        # new agents are born (between the two parents)
        vector_dx_dy = self.wrap_vector(population.pos[males] - population.pos[females])
        return (population.pos[females] + vector_dx_dy / 2) % [self.width, self.height]

    # Reproduction method for both predators and preys
    def reproduction(self):
        """Return the positions of the newborn preys and predators."""

        ### Reproduction for predators ###
        # if the number of predators is less than the maximum number of predators
        newborn_predators = np.empty((0, 2))
        if len(self.predators) < self.max_predators:
            newborn_predators = self.mating(
                self.predators,
                self.reproduction_energy_needed_predator,
                self.reproduction_energy_cost_predator,
                countdowns=(5, 2),
            )

        ### Reproduction for preys ###
        newborn_preys = self.mating(
            self.preys,
            self.reproduction_energy_needed_prey,
            self.reproduction_energy_cost_prey,
            countdowns=(3, 1),
        )
        # logistic growth: the more preys there are, the less resources for the newborns,
        # which survive with a probability 1 - (number of preys) / carrying_capacity_prey
        # (the parents pay the cost of the reproduction anyway)
        n_preys = np.sum(self.preys.energy > 0)
        survival = max(0.0, 1 - n_preys / self.carrying_capacity_prey)
        newborn_preys = newborn_preys[np.random.rand(len(newborn_preys)) < survival]

        return newborn_preys, newborn_predators

    # Remove dead agents from the world
    def remove_dead(self):

        # number of prey removed
        N_preys_removed = int(np.sum(self.preys.energy <= 0))
        self.preys.keep(self.preys.energy > 0)

        # number of predator removed
        N_predators_removed = int(np.sum(self.predators.energy <= 0))
        self.predators.keep(self.predators.energy > 0)

        return N_preys_removed, N_predators_removed

    def step(self, prey_actions=None, predator_actions=None):
        """
        Advance the simulation by one time step.

        prey_actions, predator_actions: for each agent (in the order of the population),
        how much it turns, as a proportion of max_angular_change in [-1, 1].
        If they are not given, they are computed by self.prey_policy and self.predator_policy.
        """

        # decision: all the agents decide from the same state of the world
        if prey_actions is None:
            prey_actions = self.prey_policy(self)
        if predator_actions is None:
            predator_actions = self.predator_policy(self)

        # the predators on a carcass stay immobile while they eat (and don't hunt)
        eating_carcass = self.carcass_eaten() >= 0

        # move the agents
        self.turn(self.preys, prey_actions)
        self.turn(self.predators, predator_actions)
        self.move(self.preys, self.speed_prey)
        self.move(self.predators, self.speed_predator, immobile=eating_carcass)

        # prey gain energy over time (herbivore)
        self.preys.energy = np.clip(self.preys.energy + self.energy_gain_prey, 0, self.max_energy_prey)
        # predators lose energy over time
        self.predators.energy = np.clip(self.predators.energy - self.energy_loss_predator, 0, self.max_energy_predator)

        # predators kill the preys within reach, which become carcasses
        self.hunting(can_hunt=~eating_carcass)

        # predators on a carcass eat its reserve, then the carcasses rot
        self.feeding()
        self.rotting()

        # birth
        newborn_preys, newborn_predators = self.reproduction()

        # remove dead agents
        N_preys_removed, N_predators_removed = self.remove_dead()

        # add new agents to the world
        self.spawn(self.predators, newborn_predators)
        self.spawn(self.preys, newborn_preys)

        self.time += 1

        return(
            len(self.preys),
            len(self.predators),
            len(newborn_preys),
            len(newborn_predators),
            N_preys_removed,
            N_predators_removed,
            self.mean_prey_energy,
            self.mean_predator_energy
        )
