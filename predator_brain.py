"""
Neural network that controls the predators (the "brain" of a predator), and what it perceives.

- predator_observations(world): what each predator perceives, as arrays (inputs of the network)
- PredatorBrain: the network (PyTorch), shared by all the predators
- NeuralPredatorPolicy: a policy for world.predator_policy that uses a trained PredatorBrain
- save_brain / load_brain: store a trained network in a file

The network is trained in Predator_training.ipynb.
"""
import numpy as np
import torch
from torch import nn

# maximum number of neighbours of each kind given to the network (the closest ones within the perception radius)
K_PREYS, K_PREDATORS, K_CARCASSES = 8, 6, 4

# number of features of the predator itself and of each neighbour (see predator_observations)
SELF_FEATURES, PREY_FEATURES, PREDATOR_FEATURES, CARCASS_FEATURES = 8, 5, 6, 5

# largest countdown of a predator after a birth (see World.reproduction), used to normalize it
MAX_COUNTDOWN = 5


def nearest_within(world, pos_from, pos_to, k, radius, same_agents=False):
    """
    For each position of pos_from, the k closest positions of pos_to within radius (toric world).

    return:
        index (n, k): index in pos_to of each neighbour, sorted by distance
        dx, dy (n, k): vector from pos_from to the neighbour
        found (n, k): False for the empty slots (fewer than k neighbours within radius)
    """
    n, m = len(pos_from), len(pos_to)
    index = np.zeros((n, k), dtype=int)
    dx, dy = np.zeros((n, k)), np.zeros((n, k))
    found = np.zeros((n, k), dtype=bool)
    if n == 0 or m == 0:
        return index, dx, dy, found

    all_dx, all_dy = world.displacement(pos_from, pos_to)
    distance = np.hypot(all_dx, all_dy)
    if same_agents:
        np.fill_diagonal(distance, np.inf)
    distance[distance > radius] = np.inf

    # the k smallest distances of each row (argpartition is faster than sorting the whole row)
    kk = min(k, m)
    closest = np.argpartition(distance, kk - 1, axis=1)[:, :kk]
    order = np.take_along_axis(distance, closest, axis=1).argsort(axis=1)
    closest = np.take_along_axis(closest, order, axis=1)

    index[:, :kk] = closest
    dx[:, :kk] = np.take_along_axis(all_dx, closest, axis=1)
    dy[:, :kk] = np.take_along_axis(all_dy, closest, axis=1)
    found[:, :kk] = np.isfinite(np.take_along_axis(distance, closest, axis=1))
    return index, dx, dy, found


def predator_observations(world):
    """
    What each predator perceives, as float32 arrays (one row per predator):

    self (n, 8): position in the toric world (sin and cos of x and y), direction (sin, cos),
                 energy / max energy, countdown after a birth / its maximum
    preys (n, K_PREYS, 5), predators (n, K_PREDATORS, 6), carcasses (n, K_CARCASSES, 5):
        the closest neighbours within the perception radius, seen from the predator:
        position in front of it and on its left, distance (all divided by the perception radius),
        and for the preys and predators their direction relative to the predator (sin, cos),
        the energy of the predators, and for the carcasses the energy left and the energy
        that this predator can still take from it (see World.carcass_allowance)
    *_mask: True where there is a neighbour (the other slots are empty and filled with 0)
    """
    preys, predators, carcasses = world.preys, world.predators, world.carcasses
    radius = world.perception_predator
    theta = predators.direction
    cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]

    def local(dx, dy):
        """Vector seen from the predator: (in front, on the left), divided by the perception radius."""
        return (dx * cos + dy * sin) / radius, (-dx * sin + dy * cos) / radius

    # the predator itself
    x = 2 * np.pi * predators.x / world.width
    y = 2 * np.pi * predators.y / world.height
    own = np.column_stack([
        np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(theta), np.cos(theta),
        predators.energy / world.max_energy_predator,
        predators.countdown / MAX_COUNTDOWN,
    ])

    # preys within reach
    index, dx, dy, prey_mask = nearest_within(world, predators.pos, preys.pos, K_PREYS, radius)
    front, left = local(dx, dy)
    relative = preys.direction[index] - theta[:, None] if len(preys) else np.zeros_like(dx)
    prey_features = np.stack([front, left, np.hypot(front, left), np.sin(relative), np.cos(relative)], axis=-1)

    # other predators within reach, with their energy
    index, dx, dy, predator_mask = nearest_within(
        world, predators.pos, predators.pos, K_PREDATORS, radius, same_agents=True
    )
    front, left = local(dx, dy)
    relative = predators.direction[index] - theta[:, None] if len(predators) else np.zeros_like(dx)
    energy = predators.energy[index] / world.max_energy_predator if len(predators) else np.zeros_like(dx)
    predator_features = np.stack(
        [front, left, np.hypot(front, left), np.sin(relative), np.cos(relative), energy], axis=-1
    )

    # carcasses within reach, with the energy left and the energy this predator can still take
    index, dx, dy, carcass_mask = nearest_within(world, predators.pos, carcasses.pos, K_CARCASSES, radius)
    front, left = local(dx, dy)
    if len(carcasses):
        energy = carcasses.energy[index] / world.max_energy_prey
        allowance = np.take_along_axis(world.carcass_allowance(), index, axis=1) / world.max_energy_prey
    else:
        energy, allowance = np.zeros_like(dx), np.zeros_like(dx)
    carcass_features = np.stack([front, left, np.hypot(front, left), energy, allowance], axis=-1)

    # the empty slots are filled with 0
    prey_features[~prey_mask] = 0
    predator_features[~predator_mask] = 0
    carcass_features[~carcass_mask] = 0

    return {
        "self": own.astype(np.float32),
        "preys": prey_features.astype(np.float32), "preys_mask": prey_mask,
        "predators": predator_features.astype(np.float32), "predators_mask": predator_mask,
        "carcasses": carcass_features.astype(np.float32), "carcasses_mask": carcass_mask,
    }


def concat_observations(observations):
    """Concatenate the observations of several worlds (along the predators)."""
    return {key: np.concatenate([obs[key] for obs in observations]) for key in observations[0]}


def to_tensors(observations, device):
    """Observations (numpy arrays) -> dict of torch tensors on the device."""
    return {key: torch.as_tensor(value, device=device) for key, value in observations.items()}


def actions_to_world(raw_actions):
    """
    Raw outputs of the network (n, 2) -> actions given to World.step (n, 2):
    turn in [-1, 1] and speed in [0, 1] (proportion of speed_predator).
    """
    raw_actions = np.asarray(raw_actions, dtype=float)
    turn = np.clip(raw_actions[:, 0], -1, 1)
    speed = np.clip((raw_actions[:, 1] + 1) / 2, 0, 1)
    return np.column_stack([turn, speed])


class SetEncoder(nn.Module):
    """
    Encodes a set of neighbours of one kind, whatever their number and their order:
    each neighbour goes through the same small network, then the results are pooled
    (mean and maximum over the neighbours that exist), and the number of neighbours is added.
    """

    def __init__(self, n_features, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.output_size = 2 * hidden + 1

    def forward(self, features, mask):
        h = self.net(features)                      # (batch, k, hidden)
        m = mask.unsqueeze(-1).float()               # (batch, k, 1)
        count = m.sum(dim=1)                         # (batch, 1)
        mean = (h * m).sum(dim=1) / count.clamp(min=1)
        maximum = h.masked_fill(m == 0, -1e9).max(dim=1).values
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum, count / mask.shape[1]], dim=-1)


class PredatorBrain(nn.Module):
    """
    Network shared by all the predators.

    The observations of a predator are encoded (the predator itself, and each set of neighbours
    with a SetEncoder), then two heads use this encoding:
    - the actor gives the mean of the 2 actions (turn, speed); the actions are drawn from a normal
      distribution around this mean, with a learned standard deviation (exploration)
    - the critic estimates the value of the situation (the rewards the predator can expect),
      which is only used during the training
    """

    def __init__(self, hidden=64, trunk=256):
        super().__init__()
        # sizes of the layers, saved with the weights to rebuild the same network
        self.config = dict(hidden=hidden, trunk=trunk)
        self.own = nn.Sequential(nn.Linear(SELF_FEATURES, hidden), nn.ReLU())
        self.preys = SetEncoder(PREY_FEATURES, hidden)
        self.predators = SetEncoder(PREDATOR_FEATURES, hidden)
        self.carcasses = SetEncoder(CARCASS_FEATURES, hidden)
        size = hidden + self.preys.output_size + self.predators.output_size + self.carcasses.output_size

        self.actor = nn.Sequential(nn.Linear(size, trunk), nn.ReLU(), nn.Linear(trunk, trunk), nn.ReLU(),
                                   nn.Linear(trunk, 2))
        self.critic = nn.Sequential(nn.Linear(size, trunk), nn.ReLU(), nn.Linear(trunk, trunk), nn.ReLU(),
                                    nn.Linear(trunk, 1))
        # log of the standard deviation of the actions (the same for all situations), learned
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

        # small initial actions: the network starts close to "go straight at half speed"
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def encode(self, obs):
        return torch.cat([
            self.own(obs["self"]),
            self.preys(obs["preys"], obs["preys_mask"]),
            self.predators(obs["predators"], obs["predators_mask"]),
            self.carcasses(obs["carcasses"], obs["carcasses_mask"]),
        ], dim=-1)

    def forward(self, obs):
        """return: distribution of the actions, value of the situation"""
        h = self.encode(obs)
        mean = self.actor(h)
        distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return distribution, self.critic(h).squeeze(-1)


class NeuralPredatorPolicy:
    """
    Policy for world.predator_policy that uses a trained PredatorBrain:
        world.predator_policy = NeuralPredatorPolicy(brain)

    deterministic: True to take the mean action (no exploration), False to draw it as during the training
    """

    def __init__(self, brain, device="cpu", deterministic=True):
        self.brain = brain.to(device).eval()
        self.device = device
        self.deterministic = deterministic

    def __call__(self, world):
        if len(world.predators) == 0:
            return np.zeros((0, 2))
        obs = to_tensors(predator_observations(world), self.device)
        with torch.no_grad():
            distribution, _ = self.brain(obs)
            raw = distribution.mean if self.deterministic else distribution.sample()
        return actions_to_world(raw.cpu().numpy())


def save_brain(path, brain, **info):
    """Save the weights of the network (and any information, e.g. the training settings)."""
    torch.save({"config": brain.config, "state_dict": brain.state_dict(), "info": info}, path)


def load_brain(path, device="cpu"):
    """Load a network saved by save_brain; return (brain, info)."""
    checkpoint = torch.load(path, map_location=device)
    brain = PredatorBrain(**checkpoint["config"])
    brain.load_state_dict(checkpoint["state_dict"])
    return brain.to(device), checkpoint["info"]
