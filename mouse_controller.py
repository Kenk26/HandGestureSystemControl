"""
Mouse Controller Module
Controls system mouse and keyboard actions via PyAutoGUI.
Used by gesture_controller.py.
"""

import pyautogui
import numpy as np
import platform
import subprocess
import time


class MouseController:

    def __init__(self, screen_width=None, screen_height=None, smoothing=5):
        screen = pyautogui.size()
        self.screen_width  = screen_width  or screen[0]
        self.screen_height = screen_height or screen[1]
        self.smoothing     = smoothing
        self.prev_x        = 0
        self.prev_y        = 0

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.0   # remove built-in delay for responsiveness

        print(f"MouseController ready  |  Screen: {self.screen_width}x{self.screen_height}")

    # ── cursor ────────────────────────────────────────────────────────────────

    def move_cursor(self, x, y, smooth=True):
        """
        Move cursor to normalised position (0-1, 0-1).
        """
        screen_x = int(np.interp(x, [0, 1], [0, self.screen_width]))
        screen_y = int(np.interp(y, [0, 1], [0, self.screen_height]))

        if smooth and self.prev_x != 0:
            screen_x = int(self.prev_x + (screen_x - self.prev_x) / self.smoothing)
            screen_y = int(self.prev_y + (screen_y - self.prev_y) / self.smoothing)

        self.prev_x = screen_x
        self.prev_y = screen_y

        try:
            pyautogui.moveTo(screen_x, screen_y)
        except Exception:
            pass

    def get_cursor_position(self):
        return pyautogui.position()

    # ── clicks ────────────────────────────────────────────────────────────────

    def left_click(self):
        try:
            pyautogui.click()
        except Exception:
            pass

    def right_click(self):
        try:
            pyautogui.rightClick()
        except Exception:
            pass

    def double_click(self):
        try:
            pyautogui.doubleClick()
        except Exception:
            pass

    # ── scroll ────────────────────────────────────────────────────────────────

    def scroll(self, direction='up', amount=30):
        try:
            pyautogui.scroll(amount if direction == 'up' else -amount)
        except Exception:
            pass

    # ── keyboard ──────────────────────────────────────────────────────────────

    def press_key(self, key):
        try:
            pyautogui.press(key)
        except Exception:
            pass

    def hotkey(self, *keys):
        try:
            pyautogui.hotkey(*keys)
        except Exception:
            pass

    # ── apps / screenshots ────────────────────────────────────────────────────

    def open_application(self, app_name):
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(f"start {app_name}", shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            else:
                subprocess.Popen([app_name])
            print(f"Opening {app_name}...")
        except Exception as e:
            print(f"Error opening {app_name}: {e}")

    def take_screenshot(self, filename="screenshot.png"):
        try:
            pyautogui.screenshot().save(filename)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Screenshot error: {e}")