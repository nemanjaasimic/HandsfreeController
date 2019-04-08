import time
import pyautogui as pyg

def left_click():
    pyg.click()
    # time.sleep(5)
    print("Left click performed")

def scroll(vscroll_amount = 10, hscroll_amount = 0):
    pyg.scroll(vscroll_amount)
    pyg.hscroll(hscroll_amount)
    # time.sleep(5)
    print("Scroll performed")

def mouse_move(up_down = None, left_right = None):
    x, y = pyg.position()
    pyg.move(x+up_down, y+left_right)
    print("Mouse move performed")

def take_screenshot():
    pyg.hotkey('command', 'shift', '5')
    time.sleep(5)
    print("Screenshot taken")
