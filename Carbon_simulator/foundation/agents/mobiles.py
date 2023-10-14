from Carbon_simulator.foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicMobileAgent(BaseAgent):
    """
    A basic mobile agent represents an individual actor in the economic simulation.

    "Mobile" refers to agents of this type being able to move around in the 2D world.
    """

    name = "BasicMobileAgent"
