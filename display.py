"""Curses wrapper to manage a view window and a status bar"""
import curses
import logging
import sys
import traceback


class Window():
    def __init__(self, statusbar_height=1):
        self.__screen = None
        self.__screen = curses.initscr()

        curses.noecho()  # do not echo the key pressed
        curses.cbreak()  # no need for enter after a keypress

        self.__screen.clear()

        self.__statusbar_height = statusbar_height
        self.__height = curses.LINES - statusbar_height
        self.__width = curses.COLS - 1

        self.__main_window = curses.newwin(self.__height,
                                           curses.COLS)
        self.__statusbar = curses.newwin(statusbar_height,
                                         curses.COLS,
                                         curses.LINES - statusbar_height,
                                         0)

        self.status("")
        curses.curs_set(False)  # hide cursor

        self.__screen.refresh()
        self.__current_y = 0

    def __del__(self):
        self.close()

    def close(self):
        # Curses clean stop
        if self.__screen is not None:
            curses.nocbreak()
            curses.echo()
            curses.endwin()

    def add_line(self, line):
        if self.__current_y >= self.__height:
            log.debug("trying to write too many lines, doing nothing")
            return

        if len(line) > self.__width:
            line = line[:-1] + ">"

        self.__main_window.addstr(self.__current_y, 0, line)
        self.__current_y += 1

    def height(self):
        return self.__height

    def statusbar_height(self):
        return self.__statusbar_height

    def width(self):
        return self.__width

    def status(self, line):
        self.__statusbar.clear()
        self.__statusbar.addstr(0, 0, "> " + line)

    def clear(self):
        self.__current_y = 0
        self.__main_window.clear()

    def refresh(self):
        self.__main_window.refresh()
        self.__statusbar.refresh()

    def resize(self):
        log.debug(self.__screen.getmaxyx())

        curses.update_lines_cols()
        self.__height = curses.LINES - self.__statusbar_height
        self.__width = curses.COLS - 1

        self.__main_window.resize(self.__height, curses.COLS)

        self.__statusbar.mvwin(self.__height, 0)
        self.__statusbar.resize(self.__statusbar_height, curses.COLS)

        self.__screen.clear()
        self.__screen.refresh()

    def get_key(self):
        return self.__screen.getkey()


if __name__ == "__main__":
    import random
    import string

    logging.basicConfig(filename="curses.log",
                        filemode='w',
                        level=logging.DEBUG)
    log = logging.getLogger(__name__)
    log.info("starting")
    window = Window()

    try:
        while True:
            window.clear()

            for i in range(window.height()):
                temp_str = ''.join(random.choices(string.ascii_lowercase,
                                                  k=window.width()))
                window.add_line(temp_str)

            window.refresh()

            key = window.get_key()
            window.status(f'{key}')
            log.debug(f'{key}')

            if key.lower() == 'q':
                break

            if key.lower() == 'h':
                window.status(f'{window.height()}')

            if key == "KEY_RESIZE":
                window.resize()
                log.debug(f'new height: {window.height()}')
                window.status('resized')

    except Exception:
        window.close()
        traceback.print_exc(file=sys.stdout)
