"""
Key handler objects for managing key presses.
"""
import logging
from enum import Enum
from xirtam.core.settings import Command

LOGGER = logging.getLogger(__name__)


class KeyState(Enum):
    """
    The current state of a key.
    """

    NOT_PRESSED = 0
    PRESSED = 1
    DEBOUNCED = 2


class KeyHandler:
    """
    Handles key presses and command execution checks.
    """

    def __init__(self):
        self.key_states = {}

    def add(self, key):
        """
        Set the given key to its pressed state.
        """
        state = self.key_states.get(key)
        if state is None or state is KeyState.NOT_PRESSED:
            self.key_states[key] = KeyState.PRESSED

    def remove(self, key):
        """
        Set the given key to its not pressed state.
        """
        self.key_states[key] = KeyState.NOT_PRESSED

    def __contains__(self, control: Command):
        """
        Check if all the control keys have been pressed.
        """
        return all(self.key_states.get(key) is KeyState.PRESSED for key in control.value)


class DebouncedKeyHandler(KeyHandler):
    """
    Handles key presses and command execution checks while additionally
    debouncing keys after they have been processed once.
    """

    def __init__(self):
        super().__init__()

    def __contains__(self, control: Command):
        """
        Check if all the control keys have been pressed but have yet
        to be processed (i.e. debounced). Only process (i.e. debounce)
        the keys if all have been pressed.
        """
        for key in control.value:
            if self.key_states.get(key) is not KeyState.PRESSED:
                return False
        for key in control.value:
            self.key_states[key] = KeyState.DEBOUNCED
        return True
