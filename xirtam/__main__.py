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
# Run manager
manager = TrainerManager if parsed_args.trainer else SimulatorManager
manager(
    world_filepath=parsed_args.world_filepath,
    robot_filepath=parsed_args.robot_filepath,
    motion_filepath=parsed_args.motion_filepath,
    output_filepath=parsed_args.output_filepath,
    generate_case=parsed_args.generate_case,
).run()
