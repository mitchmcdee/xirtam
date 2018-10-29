"""
Module containing high level class for controlling policy training.
"""
import logging
from xirtam.core.settings import LOG_LEVEL
from xirtam.utils.parser import SimulationParser
from xirtam.managers.trainer import TrainerManager
from xirtam.managers.simulator import SimulatorManager

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Extract simulation arguments
parsed_args = SimulationParser().parse_args()
manager_args = {
    "world_path": parsed_args.world_path,
    "robot_path": parsed_args.robot_path,
    "motion_path": parsed_args.motion_path,
    "output_path": parsed_args.output_path,
    "model_path": parsed_args.model_path,
    "generate": parsed_args.generate,
}
# Run manager
if parsed_args.trainer:
    TrainerManager(**manager_args).run()
elif parsed_args.simulator:
    SimulatorManager(**manager_args).run()
