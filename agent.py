from __future__ import annotations
import numpy as np

class Agent:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        # energy will go down during the simulation, and if it reaches 0, the agent dies
        self._energy = 100
        self.energy_max = 150
        self.reproduction_energy_needed = 120
        self.reproduction_energy_cost = 50

        # countdown is used to pause the agent when it is giving birth or eating
        self.countdown = 0
        # countdown_cadavre is used to pause the agent when it is eated
        self.countdown_cadavre = 0

        # direction (in radians)
        self._direction = np.random.uniform(0, 2 * np.pi)

        # After each update, the agent can change its direction by a maximum of max_angular_change
        self.max_angular_change = np.pi / 8

        # scalar speed
        self.speed = 1.

    
    @property
    def energy(self):
        return self._energy
        
    
    @energy.setter
    def energy(self, value):
        self._energy = value
        if self._energy < 0:
            self._energy = 0
        elif self._energy > self.energy_max:
            self._energy = self.energy_max
    
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

        return distance, direction


############################
class Prey(Agent):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.speed = 1.0
    

    def runaway(self, world):
        """
        Detect predators in the world and change direction to run away from them.
        """
        distance_to_predators = np.inf
        # default direction is the current direction of the prey
        run_away_direction = self.direction

        for predator in world.predators:
            distance, direction = self.get_distance(predator, world)
            if distance < distance_to_predators:
                distance_to_predators = distance
                # Run away from the predator ( the opposite direction => + pi)
                run_away_direction = direction + np.pi
            
        # We update the direction of the prey
        # using the setter method of the direction property 
        # to ensure the maximum angular change is respected
        self.direction = run_away_direction


    def update(self, world):
        
        # update the direction
        self.runaway(world)
        
        # update the velocity in x and y axis
        self.set_velocity()
        
        # update the position of the prey in the world
        # if the agent is in pause (giving birth or eat)
        if self.countdown != 0 :
            self.countdown -= 1
        else:
            self.move(world)

        # prey gain energy over time (herbivore)
        self.energy += 1

#############################
class Predator(Agent):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.speed = 2.0

    def hunt(self, world):
        """
        Detect prey in the world and change direction to hunt them.
        """
        distance_to_prey = np.inf
        # default direction is the current direction of the predator
        hunt_direction = self.direction

        for prey in world.preys:
            distance, direction = self.get_distance(prey, world)
            if distance < distance_to_prey:
                distance_to_prey = distance
                # Move towards the prey
                hunt_direction = direction

        # We update the direction of the predator
        # using the setter method of the direction property 
        # to ensure the maximum angular change is respected
        self.direction = hunt_direction

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
        self.energy -= 2