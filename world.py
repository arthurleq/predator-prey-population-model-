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

        self.color_prey = "blue"
        self.color_predator = "orange"

        self.list_predator_to_add = []
        self.list_prey_to_add = []

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
            self.preys.append(Prey(rand_pos[0], rand_pos[1]))
        # if a tuple is given, we create a prey at that position
        if "position" in kwargs:
            self.preys.append(Prey(kwargs["position"][0], kwargs["position"][1]))
    
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
            self.predators.append(Predator(rand_pos[0], rand_pos[1]))
        # if a tuple is given, we create a predator at that position
        if "position" in kwargs:
            self.predators.append(Predator(kwargs["position"][0], kwargs["position"][1]))



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
                    predator.energy += 20 
                    # the predator make a pause after eating
                    predator.countdown = 3
                    # prey dies
                    prey.energy = 0


    # Reproduction method for both predators and preys
    def reproduction(self):

        for predator in self.predators:
            # if there is another predator close enough
            for other_predator in self.predators:
                if predator != other_predator:
                    # Get the distance between the two predators
                    distance = predator.get_distance(other_predator, self)[0]
                    # if the predators are close enough AND both predators have enough energy
                    if distance < 5.0 and (predator.energy > predator.reproduction_energy_needed and other_predator.energy > other_predator.reproduction_energy_needed):

                            # both predators lose some energy
                            predator.energy -= predator.reproduction_energy_cost
                            other_predator.energy -= other_predator.reproduction_energy_cost
                            # and a new predator is born
                            self.list_predator_to_add.append(((predator.x + other_predator.x) / 2, (predator.y + other_predator.y) / 2))
                            # countdown because they are exausted
                            # female
                            predator.countdown = 5
                            # male
                            other_predator.countdown = 2

        for prey in self.preys:
            # if there is another prey close enough
            for other_prey in self.preys:
                if prey != other_prey:
                    distance = prey.get_distance(other_prey, self)[0]
                    # if the preys are close enough AND both preys have enough energy
                    if distance < 5.0 and (prey.energy > prey.reproduction_energy_needed and other_prey.energy > other_prey.reproduction_energy_needed):
                            # new prey is born
                            self.list_prey_to_add.append(((prey.x + other_prey.x) / 2, (prey.y + other_prey.y) / 2))

                            # countdown because they are exausted
                            # female
                            prey.countdown = 3
                            # male
                            other_prey.countdown = 1

                            prey.energy -= prey.reproduction_energy_cost
                            other_prey.energy -= other_prey.reproduction_energy_cost
    
    # Remove dead agents from the world
    def remove_dead(self):
        # suppression of the dead (energy = 0)
        for predator in self.predators:
            if predator.energy <= 0:
                self.predators.remove(predator)

        for prey in self.preys:
            if prey.energy <= 0:
                self.preys.remove(prey)

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
        self.remove_dead()

        # add new agents to the world
        for position in self.list_predator_to_add:
            self.add_predator(position=position)
        self.list_predator_to_add = []

        for position in self.list_prey_to_add:
            self.add_prey(position=position)
        self.list_prey_to_add = []

        self.time += 1