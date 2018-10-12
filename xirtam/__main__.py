"""
Module containing high level class for controlling policy training.
"""
import sys
import logging
from xirtam.core.settings import LOG_LEVEL
from xirtam.utils.parser import SimulationParser
from xirtam.managers.trainer import TrainerManager
from xirtam.managers.simulator import SimulatorManager

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Extract simulation arguments
sim_args = SimulationParser().parse_args(sys.argv[1:])
# Run manager
manager = TrainerManager if sim_args.trainer else SimulatorManager
manager(sim_args.world_path, sim_args.robot_path, sim_args.motion_path, sim_args.output_path).run()
