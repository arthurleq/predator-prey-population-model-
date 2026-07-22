from __future__ import annotations
import numpy as np

class Agent:

    def __init__(self, x, y, world, energy=None):
        self.world = world

        self.x = x
        self.y = y

        # energy will go down during the simulation, and if it reaches 0, the agent dies
        self._energy = energy if energy is not None else np.random.uniform(1, 150)
        self._energy_max = 150
        self._reproduction_energy_needed = 120
        self._reproduction_energy_cost = 50

        # countdown is used to pause the agent when it is giving birth or eating
        self.countdown = 0
        # countdown_cadavre is used to pause the agent when it is eated
        self.countdown_cadavre = 0

        # direction (in radians)
        self._direction = np.random.uniform(0, 2 * np.pi)

        # After each update, the agent can change its direction by a maximum of max_angular_change
        self.max_angular_change = np.pi / 8

    
    @property
    def energy(self):
        return self._energy

    @property
    def energy_max(self):
        return self._energy_max

    @property
    def reproduction_energy_needed(self):
        return self._reproduction_energy_needed

    @property
    def reproduction_energy_cost(self):
        return self._reproduction_energy_cost
    
    @energy.setter
    def energy(self, value):
        self._energy = np.clip(
            value,
            0,
            self.energy_max
        )
    
    # getter of the direction of the agent
    @property
    def direction(self):
        return self._direction

    # setter of the direction of the agent
    @direction.setter
    def direction(self, value):
        value = np.clip(value, 
                        self._direction - self.max_angular_change, 
                        self._direction + self.max_angular_change)
        self._direction = value
    
    # method to change the direction of the agent by a proportion of the maximum angular change
    def change_direction(self, proportion):
        if np.abs(proportion) > 1 :
            raise ValueError("proportion must be between 0 and 1")
        angular_change = proportion * self.max_angular_change
        self.direction += angular_change

    # method to set the velocity of the agent based on its speed and direction
    def set_velocity(self):
        self.vx = self.speed * np.cos(self.direction)
        self.vy = self.speed * np.sin(self.direction)


    def move(self, world):

        # toric world
        x_limit = world.width
        y_limit = world.height
        
        self.x += self.vx
        self.y += self.vy

        # if the agent goes out of the world, it appears on the other side
        if self.x < 0:
            self.x += x_limit
        elif self.x > x_limit:
            self.x -= x_limit
        
        if self.y < 0:
            self.y += y_limit
        elif self.y > y_limit:
            self.y -= y_limit

    def get_distance(self, other_agent: Agent, world: "World"):
        """
        Calculate the distance to another agent in the toric world and the direction.

        other_agent: the other agent
        world: the world in which the agents are
        return: distance, direction (in radians)
        """
        dx = other_agent.x - self.x
        dy = other_agent.y - self.y


        # toric adjustment
        if dx > world.width / 2:
            dx -= world.width
        elif dx < -world.width / 2:
            dx += world.width

        if dy > world.height / 2:
            dy -= world.height
        elif dy < -world.height / 2:
            dy += world.height

        distance = np.sqrt(dx ** 2 + dy ** 2)
        direction = np.arctan2(dy, dx)

        return distance, direction, [dx, dy]


############################
class Prey(Agent):

    def __init__(self, x, y, world, energy=None):
        super().__init__(x, y, world, energy=energy)

    @property
    def speed(self):
        return self.world.speed_prey

    @property
    def energy_max(self):
        return self.world.max_energy_prey

    @property
    def reproduction_energy_needed(self):
        return self.world.reproduction_energy_needed_prey

    @property
    def reproduction_energy_cost(self):
        return self.world.reproduction_energy_cost_prey    

    @property
    def energy_gain(self):
        return self.world.energy_gain_prey

    def runaway(self, world, danger_distance=10):
        """
        order of priority for the prey:
        1. If there is a predator too close, the prey will run away from it
        2. If there is no predator too close, and if the prey have enought energy, the prey will look for a mate (with enough energy too)
        3. Else, the prey will keep its current direction

        """
        distance_to_predators = np.inf
        distance_to_mate = np.inf
        # default direction is the current direction of the prey
        run_away_direction = self.direction

        for predator in world.predators:
            distance, direction, _ = self.get_distance(predator, world)
            if distance < distance_to_predators:
                distance_to_predators = distance
                # Run away from the predator ( the opposite direction => + pi)
                run_away_direction = direction + np.pi

        # if the closest predator is too close or the prey haven't enought energy, 
        # the prey will run away
        if distance_to_predators < danger_distance or self.energy < self.reproduction_energy_needed:
            # We update the direction of the prey
            # using the setter method of the direction property 
            # to ensure the maximum angular change is respected
            self.direction = run_away_direction

        else:
            # it looks for a near mate (with enough energy too)
            for other_prey in world.preys:
                # first we look for a mate with enough energy to reproduce
                if other_prey.energy > other_prey.reproduction_energy_needed:
                    # then we look for the closest mate
                    distance, direction, _ = self.get_distance(other_prey, world)
                    if distance < distance_to_mate:
                        distance_to_mate = distance
                        # Move towards the mate
                        run_away_direction = direction
  
            # We update the direction of the prey
            # using the setter method of the direction property 
            # to ensure the maximum angular change is respected
            self.direction = run_away_direction


    def update(self, world):
        
        # update the direction
        self.runaway(world, danger_distance=world.distance_danger)
        
        # update the velocity in x and y axis
        self.set_velocity()
        
        # update the position of the prey in the world
        # if the agent is in pause (giving birth or eat)
        if self.countdown != 0 :
            self.countdown -= 1
        else:
            self.move(world)

        # prey gain energy over time (herbivore)
        self.energy += self.energy_gain

#############################
class Predator(Agent):

    def __init__(self, x, y, world, energy=None):
        super().__init__(x, y, world, energy=energy)

    @property
    def speed(self):
        return self.world.speed_predator

    @property
    def energy_max(self):
        return self.world.max_energy_predator
    
    @property
    def reproduction_energy_needed(self):
        return self.world.reproduction_energy_needed_predator

    @property
    def reproduction_energy_cost(self):
        return self.world.reproduction_energy_cost_predator

    @property
    def energy_gain(self):
        return self.world.energy_gain_predator

    @property
    def energy_loss(self):
        return self.world.energy_loss_predator

    def hunt(self, world):
        """
        Order of priority for the predator:
        1. If the predator has enough energy to reproduce, it looks for a mate (with enough energy too)
        2. If the predator doesn't have enough energy to reproduce, it looks for the closest prey to hunt.
        3. Else, the predator will keep its current direction
        """
        distance_to_objectif = np.inf
        # default direction is the current direction of the predator
        direction_to_objectif = self.direction

        # if the predator has enough energy to reproduce, 
        if self.energy > self.reproduction_energy_needed:
            # it looks for a near mate (with enough energy too)
            for predator in world.predators:
                # first we look for a mate with enough energy to reproduce
                if predator.energy > predator.reproduction_energy_needed:
                    # then we look for the closest mate
                    distance, direction, _ = self.get_distance(predator, world)
                    if distance < distance_to_objectif:
                        distance_to_objectif = distance
                        # Move towards the mate
                        direction_to_objectif = direction
        
        # else, it looks for the closest prey
        else:
            for prey in world.preys:
                distance, direction, _ = self.get_distance(prey, world)
                if distance < distance_to_objectif:
                    distance_to_objectif = distance
                    # Move towards the prey
                    direction_to_objectif = direction

        # We update the direction of the predator
        # using the setter method of the direction property 
        # to ensure the maximum angular change is respected
        self.direction = direction_to_objectif

    def update(self, world):
        # update the direction
        self.hunt(world)
        # update the velocity in x and y axis
        self.set_velocity()
        
        # update the position of the predator in the world
        # if the agent is in pause (giving birth or eat)
        if self.countdown != 0 :
            self.countdown -= 1
        else:
            self.move(world)

        # predators lose energy when moving
        self.energy -= self.energy_loss