import numpy as np
from typing import List
from agent import Prey, Predator

class World:

    def __init__(self, width, height):
        
        # dimensions of the world
        self.width = width
        self.height = height

        # List of agents in the world (preys and predators)
        self.preys: List[Prey] = []
        self.predators: List[Predator] = []

        # time step of the simulation
        self.time = 0

        # distance at which the agents can reproduce
        self.distance_reproduction = 1.0

        # distance at which the prey will run away from predators instead of looking for a mate
        self.distance_danger = 5.0

        # limites for the number of agents in the world
        self.max_predators = 100
        self.max_preys = 200

        # speed of the agents in the world
        self.speed_prey = 1.0
        self.speed_predator = 1.5

        # max energy of each agent
        self.max_energy_prey = 150
        self.max_energy_predator = 150

        # energy needed for reproduction
        self.reproduction_energy_needed_prey = 150
        self.reproduction_energy_needed_predator = 150

        # energy cost for reproduction
        self.reproduction_energy_cost_prey = 30
        self.reproduction_energy_cost_predator = 30

        # energy gained by a predator when it eats a prey
        self.energy_gain_predator = 20
        # energy gain by a prey at each time step
        self.energy_gain_prey = 1

        # energy lost by a predator at each time step
        self.energy_loss_predator = 1

        self.color_prey = "blue"
        self.color_predator = "orange"

        self.list_predator_to_add = []
        self.list_prey_to_add = []

    @property
    def mean_prey_energy(self):
        if not self.preys:
            return 0
        return np.mean([prey.energy for prey in self.preys])

    @property
    def mean_predator_energy(self):
        if not self.predators:
            return 0
        return np.mean([predator.energy for predator in self.predators])


    
    # method to add preys to the world
    def add_prey(self, **kwargs):
        """
        Add preys to the world. The preys can be added in two ways:
        number (int) : By specifying the number of preys to add. The preys will be added at random positions in the world.
        position (tuple) : By specifying the position of a single prey
        """
        # if an int is given, we create that many preys with random positions
        for i in range(kwargs.get("number", 1)):
            rand_pos = np.random.rand(2) * [self.width, self.height]
            self.preys.append(Prey(rand_pos[0], rand_pos[1], self))
        # if a tuple is given, we create a prey at that position
        if "position" in kwargs:
            self.preys.append(Prey(kwargs["position"][0], kwargs["position"][1], self))
    
    # method to add predators to the world
    def add_predator(self, **kwargs):
        """
        Add predators to the world. The predators can be added in two ways:
        1. By specifying the number of predators to add. The predators will be added at random positions in the world.
        2. By specifying the position of a single predator
        """
        # if an int is given, we create that many predators with random positions
        for i in range(kwargs.get("number", 1)):
            rand_pos = np.random.rand(2) * [self.width, self.height]
            self.predators.append(Predator(rand_pos[0], rand_pos[1], self))
        # if a tuple is given, we create a predator at that position
        if "position" in kwargs:
            self.predators.append(Predator(kwargs["position"][0], kwargs["position"][1], self))
        

    def remove_all_agents(self):
        """
        Remove all agents (preys and predators) from the world.
        """
        self.preys = []
        self.predators = []

    def return_dict(self):
        """
        Return a dictionary containing the positions and direction of the preys and predators in the world.
        """
        return {
            "preys": [(prey.x, prey.y, prey.direction) for prey in self.preys],
            "predators": [(predator.x, predator.y, predator.direction) for predator in self.predators]
        }

    # Method to eat a prey if a predator is close enough
    def eating(self):
        for predator in self.predators:
            for prey in self.preys:
                distance = predator.get_distance(prey, self)[0]
                # if the distance between predator and prey is less than 1.0, the predator gains energy and the prey dies
                if distance < 1.0:  # if the predator is close enough to the prey
                    # predator gains energy
                    predator.energy += predator.energy_gain
                    # the predator make a pause after eating
                    predator.countdown = 3
                    # prey dies
                    prey.energy = 0

                    # a predator can only eat one prey at a time
                    break  


    # Reproduction method for both predators and preys
    def reproduction(self):

        ### Reproduction for predators ###
        # if the number of predators is less than the maximum number of predators
        if len(self.predators) < self.max_predators :

            for i, predator in enumerate(self.predators):
                # if there is another predator close enough
                for other_predator in self.predators[i+1:]:
               
                        # Get the distance between the two predators
                        distance, _, vector_dx_dy = predator.get_distance(other_predator, self)
                        # if the predators are close enough AND both predators have enough energy
                        if distance < self.distance_reproduction and (predator.energy >= predator.reproduction_energy_needed and other_predator.energy >= other_predator.reproduction_energy_needed):

                                # both predators lose some energy
                                predator.energy -= predator.reproduction_energy_cost
                                other_predator.energy -= other_predator.reproduction_energy_cost
                                # and a new predator is born
                                x = (predator.x + vector_dx_dy[0] / 2) % self.width
                                y = (predator.y + vector_dx_dy[1] / 2) % self.height
                                self.list_predator_to_add.append((x, y))
                                # countdown because they are exausted
                                # female
                                predator.countdown = 5
                                # male
                                other_predator.countdown = 2

                                break

        ### Reproduction for preys ###
        # if the number of preys is less than the maximum number of preys
        if len(self.preys) < self.max_preys :

            for i, prey in enumerate(self.preys):
                # if there is another prey close enough
                for other_prey in self.preys[i+1:]:

                    distance, _, vector_dx_dy = prey.get_distance(other_prey, self)
                    # if the preys are close enough AND both preys have enough energy
                    if distance < self.distance_reproduction and (prey.energy >= prey.reproduction_energy_needed and other_prey.energy >= other_prey.reproduction_energy_needed):
                                # new prey is born (between the two parents)
                                x = (prey.x + vector_dx_dy[0] / 2) % self.width
                                y = (prey.y + vector_dx_dy[1] / 2) % self.height
                                self.list_prey_to_add.append((x, y))

                                # countdown because they are exausted
                                # female
                                prey.countdown = 3
                                # male
                                other_prey.countdown = 1

                                prey.energy -= prey.reproduction_energy_cost
                                other_prey.energy -= other_prey.reproduction_energy_cost

                                break
    
    # Remove dead agents from the world
    def remove_dead(self):

        # number of prey removed
        N_preys_removed = len(self.preys) - sum(prey.energy > 0 for prey in self.preys)

        self.preys = [
            prey
            for prey in self.preys
            if prey.energy > 0
        ]
        
        # number of predator removed
        N_predators_removed = len(self.predators) - sum(predator.energy > 0 for predator in self.predators)

        self.predators = [
            predator
            for predator in self.predators
               if predator.energy > 0
        ]

        return N_preys_removed, N_predators_removed

    def step(self):
        
        # move the agents
        for prey in self.preys:
            prey.update(self)

        for predator in self.predators:
            predator.update(self)

        # eating
        self.eating()

        # birth
        self.reproduction()

        # remove dead agents
        N_preys_removed, N_predators_removed = self.remove_dead()

        N_predators_added = len(self.list_predator_to_add)
        N_preys_added = len(self.list_prey_to_add)

        # add new agents to the world
        for position in self.list_predator_to_add:
            self.add_predator(position=position)
        self.list_predator_to_add = []

        for position in self.list_prey_to_add:
            self.add_prey(position=position)
        self.list_prey_to_add = []

        self.time += 1

        return(
            len(self.preys),
            len(self.predators),
            N_preys_added,
            N_predators_added,
            N_preys_removed,
            N_predators_removed,
            self.mean_prey_energy,
            self.mean_predator_energy
        )