import argparse
import os
import time

import cv2


def download(url, path=None):
    try:
        import youtube_dl
    except ImportError as e:
        print("Can not find the module youtube_dl. Please install it by pip before using the code.")
        raise e

    ydl_opts = {"outtmpl": "%(id)s%(ext)s" if path is None else path}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def rgb_to_256(rgb):
    """
    Reference:
    1. https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    2. https://github.com/hit9/img2txt/blob/b54bf0cc9ac274a7fa738e06b534ec7975ad9c18/ansi.py#L9-L15

    ESC[38;5;⟨n⟩m Select foreground color
    ESC[48;5;⟨n⟩m Select background color
      0-  7:  standard colors (as in ESC [ 30–37 m)
      8- 15:  high intensity colors (as in ESC [ 90–97 m)
     16-231:  6 × 6 × 6 cube (216 colors): 16 + 36 × r + 6 × g + b (0 ≤ r, g, b ≤ 5)
    232-255:  grayscale from black to white in 24 steps
    """
    color_code = 16
    base_code = 36
    for x in rgb:
        color_code += int(round((x / 255.0) * 5)) * base_code
        base_code /= 6
    return int(color_code)


def resize_image(image, size, keep_ratio=False):
    height, width = image.shape[:2]
    if isinstance(size, int) and keep_ratio:
        height = int(height / max(height, width) * size)
        width = int(width / max(height, width) * size)
    elif isinstance(size, (tuple, list)) and len(size) == 2 and not keep_ratio:
        height, width = size
    else:
        raise ValueError(f"size: {size} is invalid!")
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)


def need_to_skip(fps, last_time, max_fps=None):
    to_skip = False
    interval_time = 1 / fps
    if max_fps:
        interval_time = max(interval_time, 1 / max_fps)

    curr_interval = time.time() - last_time
    if curr_interval < interval_time:
        time.sleep(max(interval_time - curr_interval, 0))
        # time.sleep(0.2)
    else:
        to_skip = True
    return to_skip


def process_video(path, convert_func):
    try:
        import curses
    except ImportError as e:
        print("We use the builtin module curses to show the video. But we can find it now.")
        raise e

    def plot_frame(stdscr):
        curses.use_default_colors()
        if not curses.has_colors():
            return

        for i in range(0, curses.COLORS):
            curses.init_pair(i + 1, i, -1)

            # Clear screen
        curses.curs_set(0)
        stdscr.clear()

        y_max, x_max = stdscr.getmaxyx()  # the value of first getting is y,not x
        win = curses.newwin(y_max, x_max, 0, 0)
        win.nodelay(True)

        capture = cv2.VideoCapture(path)
        fps = capture.get(cv2.CAP_PROP_FPS)

        last_time = time.time()
        while True:
            key = win.getch()
            if key == ord("q") or key == 27:  # q or esc
                break
            elif key == ord(" "):
                while win.getch() != key:
                    pass
                last_time = time.time()

            return_code, frame = capture.read()
            if not return_code:
                break

            if not need_to_skip(fps=fps, last_time=last_time):
                image = resize_image(image=frame, size=min(y_max, x_max), keep_ratio=True)

                h, w, c = image.shape
                for h_idx in range(h):
                    for w_idx in range(w):
                        win.addstr(h_idx, w_idx, "#", curses.color_pair(convert_func(image[h_idx, w_idx])))

                last_time = time.time()

        capture.release()

    curses.wrapper(plot_frame)


def process_image(path, template, convert_func):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # https://docs.python.org/3/library/shutil.html#querying-the-size-of-the-output-terminal >3.3
    terminal_size = os.get_terminal_size()
    image = resize_image(image, size=(terminal_size.columns, terminal_size.lines))
    h, w, c = image.shape

    for h_idx in range(h):
        for w_idx in range(w):
            print(template.format(n=convert_func(image[h_idx, w_idx]), msg="%"), end="")
        print()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="The path saving your image or video.")
    parser.add_argument("--mode", type=str, choices=["image", "video"], default="image")
    parser.add_argument("--url", type=str, help="The url to download the video via `youtube_dl`.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    assert os.environ["TERM"] == "tmux-256color"

    if args.mode == "video":
        if args.url and args.url.startswith("http"):
            download(args.url, args.path)
        process_video(path=args.path, convert_func=rgb_to_256)
    else:
        assert os.path.isfile(args.path)
        template = "\033[38;5;{n}m{msg}\033[0m"
        process_image(path=args.path, template=template, convert_func=rgb_to_256)


if __name__ == "__main__":
    main()
