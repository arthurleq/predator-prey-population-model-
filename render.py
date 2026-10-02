"""
Render a recorded simulation as an MP4 video.

frames: dict t -> World.return_dict(), the snapshot of the world after each time step
history: dict t -> tuple returned by World.step(), used for the population and energy curves

Writing the video requires imageio-ffmpeg (pip install imageio-ffmpeg), which provides ffmpeg.
"""
import os

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tqdm import tqdm

# shape of an agent: an arrow head pointing along the x axis, in units of the agent size
ARROW = np.array([[1.0, 0.0], [-0.7, 0.6], [-0.35, 0.0], [-0.7, -0.6]])

# size of the figure in inches: the size of the video in pixels is FIGSIZE * dpi
# (e.g. 960 x 480 pixels for dpi=75; the codec expects even numbers of pixels)
FIGSIZE = (12.8, 6.4)


def arrows(pos, direction, size):
    """Vertices (n, 4, 2) of the arrows of agents at positions pos (n, 2), pointing towards direction (n,)."""
    cos, sin = np.cos(direction), np.sin(direction)
    # rotation of each agent (transposed, because the points of the arrow are rows)
    rotation = np.stack([np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)], axis=-2)
    return pos[:, None, :] + size * ARROW @ rotation


class Renderer:
    """
    Matplotlib figure of the simulation: the world on the left, the curves on the right.
    draw() updates it for one time step and returns the image.
    """

    def __init__(
        self,
        history,
        world_width,
        world_height,
        max_energy_predator=150,
        color_prey="blue",
        color_predator="orange",
        bg_color="#ebffe4",
        carcass_color="#8c4614",
        prey_size=1.2,
        predator_size=2.2,
        dpi=75,
    ):
        self.prey_size = prey_size
        self.predator_size = predator_size

        # the figure is drawn off-screen (no window, nothing displayed in the notebook)
        self.fig = Figure(figsize=FIGSIZE, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        grid = self.fig.add_gridspec(
            2, 2, width_ratios=[1.2, 1],
            left=0.01, right=0.93, bottom=0.1, top=0.93, wspace=0.25, hspace=0.4
        )

        # --- world (left) ---
        ax = self.fig.add_subplot(grid[:, 0])
        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_height)
        ax.set_aspect("equal")
        ax.set_facecolor(bg_color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.title = ax.set_title("", fontsize=11)

        # carcasses: the size of the dot shows the energy left
        self.carcasses = ax.scatter(np.empty(0), np.empty(0), color=carcass_color, alpha=0.8, linewidths=0, zorder=1)

        # preys and predators: arrows pointing in their direction
        self.preys = PolyCollection([], facecolors=color_prey, edgecolors="none", zorder=2)
        self.predators = PolyCollection([], edgecolors="black", linewidths=0.4, zorder=3)
        ax.add_collection(self.preys)
        ax.add_collection(self.predators)

        # predators are colored by their energy (the palest colors are left out, to stay visible on the background)
        self.cmap = ListedColormap(colormaps["YlOrRd"](np.linspace(0.3, 1, 256)))
        self.norm = Normalize(vmin=0, vmax=max_energy_predator)
        self.fig.colorbar(ScalarMappable(self.norm, self.cmap), ax=ax, fraction=0.04, pad=0.01, label="Predator energy")

        # circle around the predators eating a carcass (under the predator, so that its arrow stays visible)
        self.eating = ax.scatter(
            np.empty(0), np.empty(0), s=500, facecolors="none", edgecolors="black", linewidths=1.2, zorder=2.5
        )

        self.fig.legend(
            handles=[
                Line2D([], [], ls="", marker=">", color=color_prey, label="Prey"),
                Line2D([], [], ls="", marker=">", markersize=9, color=self.cmap(0.7), markeredgecolor="black",
                       label="Predator"),
                Line2D([], [], ls="", marker="o", color=carcass_color, label="Carcass (size = energy left)"),
                Line2D([], [], ls="", marker="o", markersize=12, markerfacecolor="none", color="black",
                       label="Predator eating"),
            ],
            loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, fontsize=9, frameon=False,
        )

        # --- curves (right) ---
        times = sorted(history)
        # columns: see World.step (0: preys, 1: predators, 6: mean prey energy, 7: mean predator energy)
        values = np.array([history[t] for t in times], dtype=float)

        ax_pop = self.fig.add_subplot(grid[0, 1])
        ax_pop.plot(times, values[:, 0], color=color_prey)
        ax_pop.set_ylabel("Preys", color=color_prey)
        ax_pop.tick_params(axis="y", labelcolor=color_prey)
        ax_pop.set_title("Population over time", fontsize=11)
        ax_pred = ax_pop.twinx()
        ax_pred.plot(times, values[:, 1], color=color_predator)
        ax_pred.set_ylabel("Predators", color=color_predator)
        ax_pred.tick_params(axis="y", labelcolor=color_predator)
        ax_pred.set_ylim(bottom=0)

        ax_energy = self.fig.add_subplot(grid[1, 1], sharex=ax_pop)
        ax_energy.plot(times, values[:, 6], color=color_prey, label="Preys")
        ax_energy.plot(times, values[:, 7], color=color_predator, label="Predators")
        ax_energy.set_ylabel("Mean energy")
        ax_energy.set_xlabel("Time")
        ax_energy.set_title("Mean energy over time", fontsize=11)
        ax_energy.legend(fontsize=8, loc="lower right")

        # cursors showing the current time step
        self.cursors = [ax_pop.axvline(times[0], color="gray", lw=1), ax_energy.axvline(times[0], color="gray", lw=1)]

        # only the artists that change are redrawn at each time step, on top of a fixed background (blitting):
        # much faster than redrawing the whole figure
        # (the frame of the world is redrawn too, so that it stays on top of the agents crossing the border)
        self.dynamic = [
            self.carcasses, self.preys, self.eating, self.predators, *ax.spines.values(), self.title, *self.cursors
        ]
        for artist in self.dynamic:
            artist.set_animated(True)
        # the background is captured at the first draw (after any change of the axes)
        self.background = None

    def draw(self, t, frame):
        """Draw the snapshot of time step t, and return the image as an array (height, width, 4) of RGBA bytes."""
        preys, predators, carcasses = frame["preys"], frame["predators"], frame["carcasses"]

        self.preys.set_verts(arrows(preys["pos"], preys["direction"], self.prey_size))
        self.predators.set_verts(arrows(predators["pos"], predators["direction"], self.predator_size))
        self.predators.set_facecolor(self.cmap(self.norm(predators["energy"])))
        self.eating.set_offsets(predators["pos"][predators["eating"]])
        self.carcasses.set_offsets(carcasses["pos"])
        self.carcasses.set_sizes(10 + 0.5 * carcasses["energy"])

        self.title.set_text(
            f"t = {t}    preys: {len(preys['pos'])}    predators: {len(predators['pos'])}    "
            f"carcasses: {len(carcasses['pos'])}"
        )
        for cursor in self.cursors:
            cursor.set_xdata([t, t])

        if self.background is None:
            # draws everything except the animated artists
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.fig.bbox)
        self.canvas.restore_region(self.background)
        for artist in self.dynamic:
            self.fig.draw_artist(artist)
        return np.asarray(self.canvas.buffer_rgba())


def render_video(frames, history, world_width, world_height, output_path="./plot/world.mp4", fps=30, every=1,
                 quality=4, **style):
    """
    Draw the simulation and write it as an MP4 video, one image at a time (the memory used stays constant).

    every: draw one time step out of `every` (e.g. 2 makes the video twice shorter, and the file twice lighter)
    quality: quality of the compression, from 0 (smallest file) to 10 (best image);
             below 4, blotches appear around the agents
    style: options of Renderer (dpi = resolution, max_energy_predator, colors, size of the agents)
    return: the path of the video
    """
    # imported here, so that Renderer can be used without ffmpeg (e.g. to save a single image)
    import imageio_ffmpeg

    renderer = Renderer(history, world_width, world_height, **style)

    # the output folder (./plot/) is ignored by git, so it may not exist yet
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        output_path, renderer.canvas.get_width_height(), pix_fmt_in="rgba", fps=fps, codec="libx264",
        quality=quality, macro_block_size=2,
    )
    writer.send(None)  # start the writer
    try:
        for t in tqdm(sorted(frames)[::every], desc="Video"):
            writer.send(renderer.draw(t, frames[t]))
    finally:
        writer.close()

    return output_path
