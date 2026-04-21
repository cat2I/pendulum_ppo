import mujoco
from mujoco import mjx
import jax

print ("Version:", mujoco.__version__)
print ("JAX devices:", jax.devices())