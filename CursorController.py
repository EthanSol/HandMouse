from enum import Enum
import pyautogui

class CursorAction(Enum):
    NoAction = 0
    MoveCursor = 1
    LeftClick = 2
    RightClick = 3

class CursorController:
    def __init__(self, sensitivity=2):
        self.last_action = CursorAction.NoAction
        self.last_position = (0, 0)
        self.sensitivity = sensitivity

    def move_cursor(self, x_move, y_move):
        width, height = pyautogui.size()
        x_move = x_move * width * self.sensitivity
        y_move = y_move * height * self.sensitivity

        x, y = pyautogui.position()
        pyautogui.moveTo(x + x_move, y + y_move)

    def left_click(self):
        pyautogui.click(button='left')

    def right_click(self):
        pyautogui.click(button='right')

    def update_cursor(self, action: CursorAction, x_pos, y_pos):
        if action == CursorAction.LeftClick:
            if self.last_action != CursorAction.LeftClick:
                self.left_click()
                self.last_action = CursorAction.LeftClick
        elif action == CursorAction.RightClick:
            if self.last_action != CursorAction.RightClick:
                self.right_click()
                self.last_action = CursorAction.RightClick
        elif action == CursorAction.MoveCursor:
            self.move_cursor(x_pos - self.last_position[0], y_pos - self.last_position[1])
        elif action == CursorAction.NoAction: # want to avoid cycling through NoAction
            self.last_position = (x_pos, y_pos)
            return

        self.last_action = action
        self.last_position = (x_pos, y_pos)

