import sys
import time

class AnsiCodeHelper:
    ansi_codes = {
        'fg_colors': {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bright_black': '\033[1;30m',
            'bright_red': '\033[1;31m',
            'bright_green': '\033[1;32m',
            'bright_yellow': '\033[1;33m',
            'bright_blue': '\033[1;34m',
            'bright_magenta': '\033[1;35m',
            'bright_cyan': '\033[1;36m',
            'bright_white': '\033[1;37m',
            'reset_color': '\033[39m',
        },
        'bg_colors': {
            'black': '\033[40m',
            'red': '\033[41m',
            'green': '\033[42m',
            'yellow': '\033[43m',
            'blue': '\033[44m',
            'magenta': '\033[45m',
            'cyan': '\033[46m',
            'white': '\033[47m',
            'reset_background': '\033[49m',
        },
        'text_format': {
            'bold': '\033[1m',
            'underline': '\033[4m',
            'italic': '\033[3m',
            'inverse': '\033[7m',
            'blink': '\033[5m',
            'hidden': '\033[8m',
            'strike_through': '\033[9m',
            'frame': '\033[51m',
            'encircled': '\033[52m',
            'overlined': '\033[53m',
            'reset_format': '\033[0m',
        },
        'link': {
            'url_start': '\033]8;;',
            'url_end': '\033]8;;\033\\',
        },
        'utility': {
            'escape_sequence_start': '\033]',
            'escape_sequence_end': '\033\\',
            'enable_alternate_screen_buffer': '\033[?1049h',
            'disable_alternate_screen_buffer': '\033[?1049l',
        }
    }

    @staticmethod
    def get_ansi_code(category, name):
        return AnsiCodeHelper.ansi_codes.get(category, {}).get(name, '')


def ansi_link_str(url, link_text="", link_color='bright_green'):
    # Get the ANSI escape sequence for the default color
    # Check if link text is empty
    if link_text == '':
        link_text = url

    # Check if stdout is a terminal
    if sys.stdout.isatty():
        # get ansi codes
        # default_color_code = AnsiCodeHelper.get_ansi_code('fg_colors', link_color)
        # format_reset_code = AnsiCodeHelper.get_ansi_code('text_format', 'reset_format')
        start_sequence = AnsiCodeHelper.get_ansi_code('utility', 'escape_sequence_start')
        end_sequence = AnsiCodeHelper.get_ansi_code('utility', 'escape_sequence_end')
        # create url link sequence
        url_start = AnsiCodeHelper.get_ansi_code('link', 'url_start')
        url_end = AnsiCodeHelper.get_ansi_code('link', 'url_end')
        link_color_code = AnsiCodeHelper.get_ansi_code('fg_colors', link_color)
        reset_format_code = AnsiCodeHelper.get_ansi_code('text_format', 'reset_format')

        formatted_link_text = f"{url_start}{url}{end_sequence}{link_color_code}{link_text}{reset_format_code}{url_end}"

        # print(f'formatted_link_text={formatted_link_text}\n\n')

        return formatted_link_text
    else:
        # If stdout is not a terminal, return the default formatted link text
        return link_text


def ansi_color_str(s, fg='white', bg=None):
    if bg is None:
        color_type = 'fg_colors'
        color_code = AnsiCodeHelper.get_ansi_code(color_type, fg)
        reset_color_code = AnsiCodeHelper.get_ansi_code(color_type, 'reset_color')
        return f"{color_code}{s}{reset_color_code}"
    else:
        fg_color_code = AnsiCodeHelper.get_ansi_code('fg_colors', fg)
        bg_color_code = AnsiCodeHelper.get_ansi_code('bg_colors', bg)
        reset_bg_color_code = AnsiCodeHelper.get_ansi_code('bg_colors', 'reset_background')
        reset_fg_color_code = AnsiCodeHelper.get_ansi_code('fg_colors', 'reset_color')
        return f"{bg_color_code}{fg_color_code}{s}{reset_fg_color_code}{reset_bg_color_code}"


def ansi_format_str(s, format_name):
    format_code = AnsiCodeHelper.get_ansi_code('text_format', format_name)
    reset_format_code = AnsiCodeHelper.get_ansi_code('text_format', 'reset_format')
    return f"{format_code}{s}{reset_format_code}"


def ansi_rainbow_str(message, delay=0.1, iterations=10):
    # init_thread(ansi_rainbow_anim, message, delay)
    rainbow_colors = ['red', 'yellow', 'green', 'blue', 'magenta', 'cyan']
    for _ in range(iterations):
        for color in rainbow_colors:
            color_code = AnsiCodeHelper.get_ansi_code('fg_colors', color)
            reset_color_code = AnsiCodeHelper.get_ansi_code('fg_colors', 'reset_color')
            rainbow_str = ansi_color_str(message, color)
            print(rainbow_str, end='\r', flush=True)
            time.sleep(delay)
    print('\n')


def print_to_ansi_alt_screen_buffer(info):
    # Enable alternate screen buffer
    sys.stdout.write(
        f"{AnsiCodeHelper.get_ansi_code('utility', 'escape_sequence_start')}{AnsiCodeHelper.get_ansi_code('utility', 'enable_alternate_screen_buffer')}")

    # Display additional information on the alternate screen buffer
    print(info)

    # Disable alternate screen buffer
    sys.stdout.write(
        f"{AnsiCodeHelper.get_ansi_code('utility', 'escape_sequence_start')}{AnsiCodeHelper.get_ansi_code('utility', 'disable_alternate_screen_buffer')}")
