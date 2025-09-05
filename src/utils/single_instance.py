"""
Single Instance Manager for the Image Classifier application.

This module ensures only one instance of the application can run at a time.
It provides methods to detect existing instances and bring them to the foreground.
"""

import os
import sys
import logging
from typing import Optional

# Platform-specific imports
if sys.platform == "win32":
    import win32gui
    import win32con
    import win32process
    import win32api
    import win32event
else:
    import fcntl
    import psutil


class SingleInstanceManager:
    """
    Manages single instance behavior for the application.

    On Windows: Uses named mutex and window enumeration
    On other platforms: Uses file locking and process enumeration
    """

    def __init__(self, app_name: str = "ImageClassifier"):
        """
        Initialize the single instance manager.

        Args:
            app_name: Name of the application for mutex/file naming
        """
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)
        self.mutex_handle = None
        self.lock_file = None

        if sys.platform == "win32":
            self._setup_windows()
        else:
            self._setup_unix()

    def _setup_windows(self) -> None:
        """Setup Windows-specific single instance mechanism."""
        try:
            # Create named mutex
            mutex_name = f"Global\\{self.app_name}SingleInstance"
            self.mutex_handle = win32event.CreateMutex(
                None, False, mutex_name
            )

            if win32api.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
                self.logger.info("Another instance detected via mutex")
                self.mutex_handle = None
            else:
                self.logger.info("Single instance mutex created successfully")

        except Exception as e:
            self.logger.error(f"Failed to create Windows mutex: {e}")
            self.mutex_handle = None

    def _setup_unix(self) -> None:
        """Setup Unix-like platform single instance mechanism."""
        try:
            # Use app data directory for lock file
            from .file_utils import get_app_data_dir
            lock_dir = get_app_data_dir()
            lock_dir.mkdir(parents=True, exist_ok=True)

            self.lock_file = lock_dir / f"{self.app_name}.lock"

            # Try to acquire file lock
            self.lock_file_handle = open(self.lock_file, 'w')
            fcntl.flock(self.lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write current PID to lock file
            self.lock_file_handle.write(str(os.getpid()))
            self.lock_file_handle.flush()

            self.logger.info("Single instance lock file created successfully")

        except (IOError, OSError) as e:
            self.logger.info(f"Another instance detected via lock file: {e}")
            self.lock_file_handle = None
        except Exception as e:
            self.logger.error(f"Failed to create Unix lock file: {e}")
            self.lock_file_handle = None

    def is_first_instance(self) -> bool:
        """
        Check if this is the first instance of the application.

        Returns:
            True if this is the first instance, False otherwise
        """
        if sys.platform == "win32":
            return self.mutex_handle is not None
        else:
            return self.lock_file_handle is not None

    def try_switch_to_existing(self) -> bool:
        """
        Try to switch to an existing instance of the application.

        Returns:
            True if successfully switched, False otherwise
        """
        if sys.platform == "win32":
            return self._try_switch_windows()
        else:
            return self._try_switch_unix()

    def _try_switch_windows(self) -> bool:
        """Try to switch to existing Windows instance."""
        try:
            # Enumerate all windows to find our application
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd)
                    if self.app_name.lower() in window_text.lower():
                        # Get process ID of the window
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)

                        # Check if it's our application (not this process)
                        if pid != os.getpid():
                            windows.append((hwnd, window_text, pid))
                        return True
                return True

            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)

            if windows:
                hwnd, window_text, pid = windows[0]
                self.logger.info(f"Found existing instance: {window_text} (PID: {pid})")

                # Try to bring window to foreground
                try:
                    # Restore if minimized
                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

                    # Bring to front
                    win32gui.SetForegroundWindow(hwnd)

                    # Flash the window to get user's attention
                    win32gui.FlashWindow(hwnd, True)

                    self.logger.info("Successfully switched to existing instance")
                    return True

                except Exception as e:
                    self.logger.warning(f"Failed to bring window to foreground: {e}")

            return False

        except Exception as e:
            self.logger.error(f"Error trying to switch to existing Windows instance: {e}")
            return False

    def _try_switch_unix(self) -> bool:
        """Try to switch to existing Unix instance."""
        try:
            if not self.lock_file or not self.lock_file.exists():
                return False

            # Read PID from lock file
            try:
                with open(self.lock_file, 'r') as f:
                    pid_str = f.read().strip()
                    if pid_str.isdigit():
                        existing_pid = int(pid_str)

                        # Check if process is still running
                        if psutil.pid_exists(existing_pid):
                            self.logger.info(f"Found existing instance with PID: {existing_pid}")

                            # Try to find the window (this is more complex on Unix)
                            # For now, just return True to indicate another instance exists
                            return True

            except (ValueError, IOError):
                pass

            return False

        except Exception as e:
            self.logger.error(f"Error trying to switch to existing Unix instance: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources when the application exits."""
        try:
            if sys.platform == "win32":
                if self.mutex_handle:
                    win32api.CloseHandle(self.mutex_handle)
                    self.mutex_handle = None
            else:
                if hasattr(self, 'lock_file_handle') and self.lock_file_handle:
                    fcntl.flock(self.lock_file_handle.fileno(), fcntl.LOCK_UN)
                    self.lock_file_handle.close()

                    # Remove lock file
                    if self.lock_file and self.lock_file.exists():
                        self.lock_file.unlink()

            self.logger.info("Single instance manager cleaned up")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def check_single_instance(app_name: str = "ImageClassifier") -> Optional[SingleInstanceManager]:
    """
    Check if this is the first instance and return manager if so.

    Args:
        app_name: Name of the application

    Returns:
        SingleInstanceManager if this is the first instance, None otherwise
    """
    manager = SingleInstanceManager(app_name)

    if manager.is_first_instance():
        return manager
    else:
        # Try to switch to existing instance
        if manager.try_switch_to_existing():
            print(f"Another instance of {app_name} is already running.")
            print("Switching to the existing instance...")
            sys.exit(0)
        else:
            print(f"Error: Another instance of {app_name} is already running.")
            print("Unable to switch to the existing instance.")
            print("Please close the other instance and try again.")
            sys.exit(1)

    return None
