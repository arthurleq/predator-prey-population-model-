"""
Decision rules of the agents.

A policy is a function policy(world) that returns, for every agent of one species,
how much it turns during the time step, as a proportion of world.max_angular_change
in [-1, 1] (positive = counterclockwise). It only decides: the world applies these
actions and moves the agents (see World.step).

The heuristics below are the hand-written behaviours of the simulation.
Any function with the same signature (e.g. a neural network) can replace them:
    world.predator_policy = my_policy
"""
import numpy as np
from agent import wrap_angle


def turn_towards(current_direction, target_direction, max_angular_change):
    """Action in [-1, 1] that turns from the current direction towards the target, by the short way."""
    return np.clip(wrap_angle(target_direction - current_direction) / max_angular_change, -1, 1)


def closest(world, pos_from, pos_to, same_agents=False):
    """
    For each position of pos_from, find the closest position of pos_to in the toric world.

    pos_from: array (n, 2), pos_to: array (m, 2)
    same_agents: True if pos_from and pos_to are the same agents, so that an agent doesn't target itself
    return: distance (inf if there is no target), direction (in radians)
    """
    n = len(pos_from)
    if n == 0 or len(pos_to) == 0:
        return np.full(n, np.inf), np.zeros(n)

    dx, dy = world.displacement(pos_from, pos_to)
    distance = np.hypot(dx, dy)
    if same_agents:
        np.fill_diagonal(distance, np.inf)

    rows = np.arange(n)
    j = distance.argmin(axis=1)
    return distance[rows, j], np.arctan2(dy[rows, j], dx[rows, j])


def prey_heuristic(world):
    """
    Order of priority for the prey:
    1. If the closest predator is too close (world.distance_danger) or if the prey doesn't have
       enough energy to reproduce, the prey runs away from the closest predator
    2. Else, it moves towards the closest prey that has enough energy to reproduce too
    3. If there is no target, it keeps its current direction
    """
    preys, predators = world.preys, world.predators

    # default direction is the current direction of the prey
    target = preys.direction.copy()

    # run away from the closest predator (the opposite direction => + pi)
    distance_to_predator, direction_to_predator = closest(world, preys.pos, predators.pos)
    has_predator = np.isfinite(distance_to_predator)
    target[has_predator] = direction_to_predator[has_predator] + np.pi

    # if no predator is too close and the prey has enough energy, it looks for a mate instead
    can_reproduce = preys.energy >= world.reproduction_energy_needed_prey
    look_for_mate = (distance_to_predator >= world.distance_danger) & can_reproduce

    # the mates are the other preys with enough energy to reproduce
    mates = np.flatnonzero(can_reproduce)
    distance_to_mate, direction_to_mate = closest(world, preys.pos[mates], preys.pos[mates], same_agents=True)
    found = look_for_mate[mates] & np.isfinite(distance_to_mate)
    target[mates[found]] = direction_to_mate[found]

    return turn_towards(preys.direction, target, world.max_angular_change)


def predator_heuristic(world):
    """
    Order of priority for the predator:
    1. If the predator has enough energy to reproduce, it looks for the closest mate (with enough energy too)
    2. If it doesn't have enough energy to reproduce, it goes to the closest food:
       a prey to hunt or a carcass to eat (possibly shared with other predators)
    3. If there is no target, it keeps its current direction
    """
    preys, predators, carcasses = world.preys, world.predators, world.carcasses

    # default direction is the current direction of the predator
    target = predators.direction.copy()

    can_reproduce = predators.energy >= world.reproduction_energy_needed_predator

    # predators with enough energy look for the closest mate (with enough energy too)
    mates = np.flatnonzero(can_reproduce)
    distance, direction = closest(world, predators.pos[mates], predators.pos[mates], same_agents=True)
    found = np.isfinite(distance)
    target[mates[found]] = direction[found]

    # the others go to the closest food (prey or carcass)
    hunters = np.flatnonzero(~can_reproduce)
    food = np.concatenate([preys.pos, carcasses.pos])
    distance, direction = closest(world, predators.pos[hunters], food)
    found = np.isfinite(distance)
    target[hunters[found]] = direction[found]

    return turn_towards(predators.direction, target, world.max_angular_change)
