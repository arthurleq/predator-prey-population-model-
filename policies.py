"""
Decision rules of the agents.

A policy is a function policy(world) that returns, for every agent of one species,
how much it turns during the time step, as a proportion of world.max_angular_change
in [-1, 1] (positive = counterclockwise). It only decides: the world applies these
actions and moves the agents (see World.step).

An agent only perceives what is within its perception radius (world.perception_prey,
world.perception_predator): preys, predators and carcasses further away are invisible.

The heuristics below are the hand-written behaviours of the simulation.
Any function with the same signature (e.g. a neural network) can replace them:
    world.predator_policy = my_policy
"""
import numpy as np
from agent import wrap_angle


def turn_towards(current_direction, target_direction, max_angular_change):
    """Action in [-1, 1] that turns from the current direction towards the target, by the short way."""
    return np.clip(wrap_angle(target_direction - current_direction) / max_angular_change, -1, 1)


def wander(n):
    """Random turns, used when an agent perceives nothing to go to (random walk)."""
    return np.random.uniform(-1, 1, n)


def closest(world, pos_from, pos_to, radius=np.inf, same_agents=False):
    """
    For each position of pos_from, find the closest position of pos_to within radius (toric world).

    pos_from: array (n, 2), pos_to: array (m, 2)
    radius: perception radius, nothing further away is seen
    same_agents: True if pos_from and pos_to are the same agents, so that an agent doesn't target itself
    return: distance (inf if nothing is perceived), direction (in radians)
    """
    n = len(pos_from)
    if n == 0 or len(pos_to) == 0:
        return np.full(n, np.inf), np.zeros(n)

    dx, dy = world.displacement(pos_from, pos_to)
    distance = np.hypot(dx, dy)
    if same_agents:
        np.fill_diagonal(distance, np.inf)
    distance[distance > radius] = np.inf

    rows = np.arange(n)
    j = distance.argmin(axis=1)
    return distance[rows, j], np.arctan2(dy[rows, j], dx[rows, j])


def prey_heuristic(world):
    """
    Order of priority for the prey:
    1. If it perceives a predator, and this predator is too close (world.distance_danger) or the prey
       doesn't have enough energy to reproduce, the prey runs away from the closest predator
    2. Else, if it has enough energy, it moves towards the closest perceived prey that has enough energy too
       (if it perceives none, it keeps running away from the predator it perceives, if any)
    3. If there is no target, it wanders randomly
    """
    preys, predators = world.preys, world.predators
    radius = world.perception_prey

    # target direction of each prey (nan = no target)
    target = np.full(len(preys), np.nan)

    # run away from the closest perceived predator (the opposite direction => + pi)
    distance_to_predator, direction_to_predator = closest(world, preys.pos, predators.pos, radius)
    sees_predator = np.isfinite(distance_to_predator)
    target[sees_predator] = direction_to_predator[sees_predator] + np.pi

    # if no predator is too close and the prey has enough energy, it looks for a mate instead
    # (if it finds none, it keeps running away from the predator it perceives)
    can_reproduce = preys.energy >= world.reproduction_energy_needed_prey
    look_for_mate = (distance_to_predator >= world.distance_danger) & can_reproduce

    # the mates are the other preys with enough energy to reproduce
    mates = np.flatnonzero(can_reproduce)
    distance_to_mate, direction_to_mate = closest(
        world, preys.pos[mates], preys.pos[mates], radius, same_agents=True
    )
    found = look_for_mate[mates] & np.isfinite(distance_to_mate)
    target[mates[found]] = direction_to_mate[found]

    # the preys without target wander
    actions = wander(len(preys))
    has_target = ~np.isnan(target)
    actions[has_target] = turn_towards(preys.direction[has_target], target[has_target], world.max_angular_change)
    return actions


def predator_heuristic(world):
    """
    Order of priority for the predator:
    1. If the predator has enough energy to reproduce, it goes towards the closest perceived predator
       that has enough energy too
    2. If it doesn't have enough energy to reproduce, it goes to the closest perceived food:
       a prey to hunt or a carcass to eat (possibly shared with other predators)
    3. If there is no target, it wanders randomly
    """
    preys, predators, carcasses = world.preys, world.predators, world.carcasses
    radius = world.perception_predator

    # target direction of each predator (nan = no target)
    target = np.full(len(predators), np.nan)

    can_reproduce = predators.energy >= world.reproduction_energy_needed_predator

    # predators with enough energy look for the closest mate (with enough energy too)
    mates = np.flatnonzero(can_reproduce)
    distance, direction = closest(world, predators.pos[mates], predators.pos[mates], radius, same_agents=True)
    found = np.isfinite(distance)
    target[mates[found]] = direction[found]

    # the others go to the closest food (prey or carcass)
    hunters = np.flatnonzero(~can_reproduce)
    food = np.concatenate([preys.pos, carcasses.pos])
    distance, direction = closest(world, predators.pos[hunters], food, radius)
    found = np.isfinite(distance)
    target[hunters[found]] = direction[found]

    # the predators without target wander
    actions = wander(len(predators))
    has_target = ~np.isnan(target)
    actions[has_target] = turn_towards(
        predators.direction[has_target], target[has_target], world.max_angular_change
    )
    return actions
