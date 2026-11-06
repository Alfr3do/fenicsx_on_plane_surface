import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load exported array snapshots (N_steps, N_points, 2)
data = np.load("flow_snapshots.npz")
coords = data["coords"]  # (x, y)
u_frames = data["velocity"]  # (N_steps, N_points)
times = data["times"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_aspect("equal")

# Initial frame
trip = ax.tricontourf(coords[:, 0], coords[:, 1], u_frames[0], levels=50, cmap="jet")
cbar = fig.colorbar(trip, ax=ax, label="Velocity (m/s)")
title = ax.set_title(f"Velocity Field t = {times[0]:.3f}s")

def update(frame):
    ax.clear()
    ax.set_aspect("equal")
    cnt = ax.tricontourf(coords[:, 0], coords[:, 1], u_frames[frame], levels=50, cmap="jet")
    ax.set_title(f"Navier-Stokes Flow Past Cylinder (t = {times[frame]:.3f}s)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return cnt,

anim = FuncAnimation(fig, update, frames=len(times), interval=50)
anim.save("figures/flow_field.gif", writer="pillow", dpi=150)
plt.close(fig)
print("✅ Saved figures/flow_field.gif")
