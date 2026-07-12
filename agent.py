from __future__ import annotations
import numpy as np

class Agent:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.age = 0
        # maximum age for agents (preys and predators)
        self.max_age = 100

        # energy will go down during the simulation, and if it reaches 0, the agent dies
        self.energy = 100
        self.reproduction_energy_needed = 120
        self.reproduction_energy_cost = 50

        self.countdown = 0

        # direction (in radians)
        self.direction = np.random.uniform(0, 2 * np.pi)

        # After each update, the agent can change its direction by a maximum of max_angular_change
        self.max_angular_change = np.pi / 8

        # scalar speed
        self.speed = 1.

    
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
        # if the agent is in pause (giving birth or eat)
        if self.countdown != 0 :
            self.countdown -= 1
            # speed = 0
            self.x += 0
            self.y += 0
        # else, update of the velocity
        else :
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
        for predator in world.predators:
            distance, direction = self.get_distance(predator, world)
            if distance < distance_to_predators:
                distance_to_predators = distance
                # Run away from the predator
                run_away_direction = direction + np.pi
            
        # We update the direction of the prey with a maximum angular change
        diff_direction = self.direction - run_away_direction
        diff_direction = np.clip(diff_direction, -self.max_angular_change, self.max_angular_change)
        self.direction -= diff_direction


    def update(self, world):
        
        # update the direction
        self.runaway(world)
        # update the velocity in x and y axis
        self.set_velocity()
        # update the position of the prey in the world
        self.move(world)

        # prey don't lose energy when moving, but they age
        self.age += 1

#############################
class Predator(Agent):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.speed = 1.2

    def hunt(self, world):
        """
        Detect prey in the world and change direction to hunt them.
        """
        distance_to_prey = np.inf
        for prey in world.preys:
            distance, direction = self.get_distance(prey, world)
            if distance < distance_to_prey:
                distance_to_prey = distance
                # Move towards the prey
                hunt_direction = direction

        # We update the direction of the predator with a maximum angular change
        diff_direction = hunt_direction - self.direction
        diff_direction = np.clip(diff_direction, -self.max_angular_change, self.max_angular_change)
        self.direction += diff_direction

    def update(self, world):
        # update the direction
        self.hunt(world)
        # update the velocity in x and y axis
        self.set_velocity()
        # update the position of the predator in the world
        self.move(world)

        # predators lose energy when moving, and they age
        self.energy -= 2
        self.age += 1