# Standard Library Imports
import datetime
import threading
import time
import traceback
from pathlib import Path

import enum

# Third-Party Imports
from loguru import logger

# Local Imports
from .ansi_utils import ansi_color_str, ansi_link_str
from .attr_utils import get_prev_caller_info, get_caller_info, get_thread_info, is_of_type, get_call_stack_info
from .extended_symbols import ExtendedSymbols, format_arrow

# Constants
lastLog = None  # Stores the most recent log entry
debug_log = []  # Stores all log entries
startTime = time.time()
# Settings vars
enable_log = True
enable_log_to_file = False
log_filename = "log"
log_file_path = "data/log/log.txt"


enable_use_color = True

type_color = 'yellow'
key_color = 'bright_cyan'
value_color = 'bright_yellow'
arrow_color = 'bright_white'
error_color = 'bright_red'

def allow_curly_braces(original_string):
    if "{" in original_string or "}" in original_string:
        escaped_string = original_string.replace("{", "{{").replace("}", "}}")
        # print("Escaped String:", escaped_string)  # Debug output
        return escaped_string
    return original_string

def stringf(s: str, *args, **kwargs):
    if not args and not kwargs:
        return s
    else:
        return s.format(*args, **kwargs)

def format_arg(arg, use_color=False):
    def is_complex_type(obj):
        return is_of_type(obj, (list, set, dict)) or hasattr(obj, '__dict__')

    type_str = f"<{type(arg).__name__}>"
    # if is_of_type(arg, (enum)):
    #     enum_value = arg.value
    
    value_str = repr(arg) if not is_of_type(arg, str) else f'""{arg}""' 
    newline_if_needed = '\n' if is_complex_type(arg) else ""
    formatted_arg = f"{type_str}:{newline_if_needed}{value_str}"
    formatted_arg = allow_curly_braces(formatted_arg)
    return ansi_color_str(formatted_arg, fg=value_color) if use_color else formatted_arg

def format_kwarg(kw, arg, use_color=True):
    name_str = ansi_color_str(kw, fg=key_color) if use_color else kw
    arg_str = format_arg(arg, use_color)
    formatted_arg = f"{name_str}: {arg_str}"
    return ansi_color_str(formatted_arg, fg=value_color) if use_color else formatted_arg

def format_args(use_color=enable_use_color, *args, **kwargs):
    formatted_args = [format_arg(arg, use_color) for arg in args]
    formatted_kwargs = {k: format_kwarg(k, v, use_color) for k, v in kwargs.items()}
    return formatted_args, formatted_kwargs

def make_formatted_msg(msg, steps=3, use_color=True, *args, **kwargs):
    formatted_args, formatted_kwargs = format_args(use_color, *args, **kwargs)
    msgf = stringf(msg, *formatted_args, **formatted_kwargs)
    formatted_msg = f"\n ~ | {get_thread_info(get_prev_caller_info(msgf, steps=steps, use_color=use_color))}"
    return formatted_msg

def ensure_directory_exists(directory):
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error("Error occurred while ensuring directory exists", e)

def ensure_file_exists(file_path):
    try:
        path = Path(file_path)
        path.touch(exist_ok=True)
    except Exception as e:
        log_error("Error occurred while ensuring file exists", e)

def init_log(log_config):
    global enable_log_to_file, enable_log, log_filename, log_file_path
    try:
        enable_log_to_file = log_config['log_to_file']
        clear_file_first = log_config['clear_file_first']
        enable_log = log_config['enable_log']
        log_filename = log_config['log_filename']
        log_dir = 'data/log/'
        ensure_directory_exists(log_dir)
        log_file_path = f"data/log/{log_filename}.txt"
        ensure_file_exists(log_file_path)
        with open(log_file_path, 'w') as fp:
            pass
        log_info(f'init_log: \n ~ enable_log: {enable_log} \n ~ enable_log_to_file: {enable_log_to_file} \n ~ log_file_path: {log_file_path}')
    except Exception as e:
        log_error("Error occurred while initializing log", e)

log_dir = 'data/log/'
ensure_directory_exists(log_dir)
log_file_path = f"data/log/mew_log.txt"
ensure_file_exists(log_file_path)

def _update_log(formatted_msg_str):
    global lastLog, debug_log, enable_log_to_file
    try:
        lastLog = formatted_msg_str
        debug_log.append(lastLog)
        if enable_log_to_file:
            log_file(lastLog)
    except Exception as e:
        log_error("Error occurred while updating log", e)

def log_file(msg, log_file_path="data/log/mew_log.txt"):
    try:
        if not enable_log:
            return
        
        log_file_path = Path(log_file_path)
        log_dir = log_file_path.parent
        
        # Ensure the directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        #log message
        cur_time = time.time()
        elapsed_time = cur_time - startTime
        with log_file_path.open('a') as fp:  # Use 'a' to append instead of overwrite
            fp.write(f"\n ~ | time: [{format_duration(elapsed_time)}] | {msg}")
    except Exception as e:
        log_error("Error occurred while logging to file", e)

def log_info(msg, *args, **kwargs):
    try:
        if not enable_log:
            return
        formatted_msg_str = make_formatted_msg(msg, *args, **kwargs)
        _update_log(formatted_msg_str)
        formatted_log_str = lastLog
        print(formatted_log_str)
    except Exception as e:
        log_error("Error occurred while logging info", e)

def log_warning(msg, *args, **kwargs):
    try:
        if not enable_log:
            return
        formatted_msg_str = make_formatted_msg(msg, *args, **kwargs)
        _update_log(formatted_msg_str)
        formatted_log_str = lastLog
        logger.warning(formatted_log_str)
    except Exception as e:
        log_error("Error occurred while logging warning", e)

def log_error(msg, e, *args, **kwargs):
    try:
        if not enable_log:
            return
        formatted_msg_str = f"\n===^_^===\n{make_formatted_msg(msg, *args, **kwargs)}" 
        formatted_msg_str += f"{trace_error(e)}\n===>_<==="
        _update_log(formatted_msg_str)
        formatted_log_str = lastLog
        logger.error(formatted_log_str)
    except Exception as e:
        log_error("Error occurred while logging error", e)

def format_traceback(tb_entry, depth=0, use_color=enable_use_color):
    try:
        arrow_str = format_arrow('T', 'double', 3 + 2 * depth)   
        filename, lineno, funcname, line = tb_entry
        file_path = f"file:///{filename}"
        file_link = f"{filename}::{lineno}::{funcname}"
        trace_link = ansi_link_str(file_path, file_link, link_color="bright_magenta") if use_color else file_link
        arrow_str2 = format_arrow('L', 'double', 3 + 2 * (depth+1))
        formatted_tb_str = f'\n ~ | {arrow_str}trace({depth}): {trace_link}\n ~ | {arrow_str2} Line: "{line}"'
        return ansi_color_str(formatted_tb_str, fg=error_color) if use_color else formatted_tb_str
    except Exception as e:
        log_error("Error occurred while formatting traceback", e)

def trace_error(e, use_color=True, fancy=True):
    try:
        exception_str = f"\n ~ | {ExtendedSymbols.DOUBLE_VERTICAL}Exception: {repr(e)}"
        out_str = ansi_color_str(exception_str, fg=error_color) if use_color else exception_str
        if isinstance(e, BaseException):
            arrow_str = format_arrow('T', 'double', 3)
            traceback_obj = e.__traceback__
            if traceback_obj:
                file_path = f"file:///{traceback_obj.tb_frame.f_code.co_filename}"
                file_link = f"{traceback_obj.tb_frame.f_code.co_filename}::{traceback_obj.tb_lineno}"
                trace_link = ansi_link_str(file_path, file_link, link_color="bright_magenta") if use_color else file_link
                formatted_tb_str = f"\n ~ | {arrow_str}trace(0): {trace_link}"
                out_str += ansi_color_str(formatted_tb_str, fg=error_color) if use_color else formatted_tb_str
                tb = traceback.extract_tb(traceback_obj)
                for i, tb_entry in enumerate(tb):
                    out_str += format_traceback(tb_entry, depth=i+1, use_color=use_color)
        return ansi_color_str(out_str, fg=error_color) if use_color else out_str
    except Exception as e:
        log_error("Error occurred while tracing error", e)

def log_function(func):
    try:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def get_func_args():
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                func_args = dict(zip(arg_names, args))
                func_args.update(kwargs)

            try:
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                func_args = dict(zip(arg_names, args))
                func_args.update(kwargs)
                func_args_str = format_table(func_args, headers=None, tablefmt='simple')
                result = func(*args, **kwargs)
                print(get_thread_info(get_prev_caller_info(f"({func_args} ) -> result: {format_arg(result, use_color=True)}")))
                return result
            except Exception as e:
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                func_args = dict(zip(arg_names, args))
                func_args.update(kwargs)
                logger.error(f"Exception: {str(e)}")
                logger.error(f"Format Traceback: {traceback.format_exc()}")
                raise  # Re-raise the exception after logging
        return wrapper
    except Exception as e:
        log_error("Error occurred while logging function", e)

def format_duration(seconds):
    try:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        out_str = ""

        if hours > 0:
            out_str += f"{hours}h"
        if minutes > 0 or hours > 0:  # Ensure minutes show if there are hours
            out_str += f"{minutes}m"
        out_str += f"{seconds}s"  

        return out_str
    except Exception as e:
        log_error("Error occurred while formatting duration", e)
        return "???"  # Return a fail-safe placeholder instead of crashing


def parse_duration(duration_str):
    try:
        parts = duration_str.split()
        hours = int(parts[0][:-1]) if 'h' in parts[0] else 0
        minutes = int(parts[1][:-1]) if 'm' in parts[1] else 0
        seconds = int(parts[2][:-1]) if 's' in parts[2] else 0
        return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        log_error("Error occurred while parsing duration", e)

def get_timestamp():
    try:
        return datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S")
    except Exception as e:
        log_error("Error occurred while getting timestamp", e)

def format_path(directory, depth=0, max_depth=3, use_color=True):
    try:
        if max_depth is not None and depth > max_depth:
            return ""

        directory_path = Path(directory)
        if not directory_path.is_dir():
            return "The specified path is not a directory.\n"

        line_prefix = "│   " * depth + "├── "
        directory_name = directory_path.name
        if use_color:
            line_prefix = ansi_color_str(line_prefix, fg='cyan')
            directory_name = ansi_color_str(directory_name, fg='green')

        formatted_str = line_prefix + directory_name + "\n"

        sorted_items = sorted(directory_path.iterdir(), key=lambda x: (x.is_file(), x.name))
        
        for item in sorted_items:
            if item.is_dir():
                formatted_str += format_path(item, depth + 1, max_depth, use_color)
            else:
                file_line = "│   " * (depth + 1) + "├── " + item.name
                if use_color:
                    file_line = ansi_color_str(file_line, fg='white')
                formatted_str += file_line + "\n"

        return formatted_str
    except Exception as e:
        log_error("Error occurred while formatting path", e)



# cheaper mew log

import os
from datetime import datetime
import inspect

def _get_sub_log_path():
    """Determine log file name based on calling module."""
    frame = inspect.stack()[2]
    caller_path = frame.filename
    module_name = os.path.splitext(os.path.basename(caller_path))[0]
    return os.path.join("logs", f"_{module_name}_debug.log")

def reset_log_file():
    """Clear the log file at the start of the program."""
    path = _get_sub_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(path)} started at {datetime.now()}\n")

_log_reset_done = set()

def log_message(msg: str, also_print=True):
    """Log message with automatic per-file log routing."""
    path = _get_sub_log_path()
    if path not in _log_reset_done:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {os.path.basename(path)} started at {datetime.now()}\n")
        _log_reset_done.add(path)

    line = f" ~ | {msg}"
    if also_print:
        print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")



def run_unit_test():
    try:
        class Person:
            def __init__(self, name, age, city):
                self.name = name
                self.age = age
                self.city = city

        person1 = Person('John', 30, 'New York')
        person2 = Person('Alice', 25, 'Los Angeles')
        person3 = Person('Bob', 30, 'Hong Kong')
        person4 = Person('Charlie', 35, 'Shanghai')
        person5 = Person('David', 40, 'Beijing')

        data_dict = {'Name': 'John', 'Age': 30, 'City': 'New York'}
        data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_set = {'apple', 'banana', 'orange'}
        data_dict_of_lists = {'Name': ['John', 'Alice'], 'Age': [30, 25], 'City': ['New York', 'Los Angeles']}
        data_list_of_dicts = [person1.__dict__, person2.__dict__]
        data_dict_of_dicts = {'Person1': person1.__dict__, 'Person2': person2.__dict__}
        list_of_objs = [person3, person4, person5]
        data_list_of_lists = [['John', 30, 'New York'], ['Alice', 25, 'Los Angeles']]
        dict_of_objs = {'Alice': person3, 'Bob': person4, 'Charlie': person5}
        dict_of_list_of_objs = {'Group1': [person3, person4], 'Group2': [person5]}

        complex_data = {
            'data_dict': data_dict,
            'data_list': data_list,
            'data_set': data_set,
            'data_dict_of_lists': data_dict_of_lists,
            'data_list_of_dicts': data_list_of_dicts,
            'data_dict_of_dicts': data_dict_of_dicts,
            'list_of_objs': list_of_objs,
            'data_list_of_lists': data_list_of_lists,
            'dict_of_objs': dict_of_objs,
            'dict_of_list_of_objs': dict_of_list_of_objs
        }
        
        log_info("Data Dictionary: {}", data_dict)
        log_info("Data List: {}", data_list)
        log_info("Data Set: {}", data_set)
        log_info("Data Dictionary of Lists: {}", data_dict_of_lists)
        log_info("Data List of Dicts: {}", data_list_of_dicts)
        log_info("Data Dictionary of Dicts: {}", data_dict_of_dicts)
        log_info("List of Objects: {}", list_of_objs)
        log_info("Data List of Lists: {}", data_list_of_lists)
        log_info("Dictionary of Objects: {}", dict_of_objs)
        log_info("Dictionary of List of Objects: {}", dict_of_list_of_objs)
    except Exception as e:
        log_error("Error occurred while running unit test", e)

#run_unit_test()


