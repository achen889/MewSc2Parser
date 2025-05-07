
import threading
import inspect
import os
import traceback
import inspect

from .ansi_utils import ansi_link_str, ansi_color_str

def get_type_name(attr):
    return type(attr).__name__

def get_class_name(obj):
    return obj.__class__.__name__

def is_of_class(obj, cls):
    if cls is None:
        return False
    if isinstance(cls, str):
        return get_class_name(obj) == cls
    elif hasattr(cls, '__name__'):
        return get_class_name(obj) == cls.__name__
    else:
        return False

def is_of_type(val, types=None, raise_err=False):
    if types is None:
        types = (types,)

    if not isinstance(val, types):
        if raise_err:
            raise TypeError(f"Value must be one of {types}")
        return False
    else:
        return True

def get_partial_argspec(method):
    if not callable(method):
        return None  # Not a callable object
    try:
        full_argspec = inspect.getfullargspec(method)
        return full_argspec
    except TypeError:
        # Fallback to using inspect.signature
        signature = get_signature(method)
        if signature:
            parameters = signature.parameters
            args = [param for param in parameters if parameters[param].default == parameters[param].empty]
            varargs = signature.varargs
            varkw = signature.varkw
            defaults = [parameters[param].default for param in args]
            return inspect.FullArgSpec(args, varargs, varkw, defaults)

def get_signature(method):
    try:
        signature = inspect.signature(method)
        return signature
    except (TypeError, ValueError):
        return None

def get_method_args(method):
    full_arg_spec = get_partial_argspec(method)
    if full_arg_spec:
        args = [arg for arg in full_arg_spec.args if
                getattr(method, arg, None) is not None and getattr(method, arg, "") != ""]
        kwargs = {key: getattr(method, key, None) for key in full_arg_spec.kwonlyargs}
        kwargs_defaults = {key: value for key, value in
                           zip(full_arg_spec.kwonlyargs, full_arg_spec.kwonlydefaults or ())}
        args.extend(f"{key}={value}" for key, value in kwargs.items() if value is not None and value != "")
        return args
    return None

def get_source_info(attr):
    try:
        source_lines, line_number = inspect.getsourcelines(attr)
        source_file_path = inspect.getsourcefile(attr)
        source_file = os.path.relpath(source_file_path)
        return f"/{source_file}::{line_number}"
    except Exception as e:
        return "Source info not available!"

def get_method_info(method):
    args = get_method_args(method)
    args_str = ", ".join(args) if args else ''
    signature = get_signature(method)
    return_str = f' -> {signature.return_annotation}' if signature and signature.return_annotation is not inspect.Signature.empty else ''

    try:
        source_info = get_source_info(method)
    except Exception as e:
        raise Exception("Source info not available!", e)

    # Construct the file:// URL with line number for the method if available
    method_file_url = f"file://{inspect.getsourcefile(method)}#L{inspect.getsourcelines(method)[1]}"
    method_link = ansi_link_str(method_file_url, "Source")

    # Include the link in the method signature string
    method_signature = f"{signature}{return_str}: {method_link}\n-->{source_info}"

    return method_signature

def get_variable_info(variable):
    return f"<{get_type_name(variable)}>: {get_variable_value(variable)}"

def get_variable_value(variable):
    return f'{str(variable)}' if not is_of_type(variable, (list, set, dict)) or hasattr(variable, '__dict__') else '...'

def format_class_attributes(cls, verbose=True):
    try:
        attributes_dict = get_class_attributes(cls, verbose)
        variables_str = '\n'.join([f' - {attr}' for attr in attributes_dict['variables'].values()])
        methods_str = '\n'.join([f' - {attr}' for attr in attributes_dict['methods'].values()])
        return f'===list_class_attributes of: {cls}:\n===<variables>===\n{variables_str}\n===<methods>===\n{methods_str}'
    except Exception as e:
        print(ansi_color_str(f"Error occurred while formatting class attributes: {e}", fg='red'))

def format_class_attributes_color(cls, verbose=True):
    try:
        attributes_dict = get_class_attributes_color(cls, verbose)
        variables_str = '\n'.join([f' - {attr}' for attr in attributes_dict['variables'].values()])
        methods_str = '\n'.join([f' - {attr}' for attr in attributes_dict['methods'].values()])
        return f'===list_class_attributes of: {ansi_color_str(cls.__name__, fg="cyan")}:\n===<variables>===\n{variables_str}\n===<methods>===\n{methods_str}'
    except Exception as e:
        print(ansi_color_str(f"Error occurred while formatting class attributes: {e}", fg='red'))

def get_class_attributes(cls, verbose=True):
    attributes_dict = {'variables': {}, 'methods': {}}  # 'name': cls.__class__.__name__,

    try:
        for attribute_v in vars(cls):
            if not attribute_v.startswith('__') and 'stat' not in attribute_v:
                attr = getattr(cls, attribute_v)
                if not callable(attr):
                    if verbose:
                        attributes_dict['variables'][attribute_v] = f' ~ {attribute_v}{get_variable_info(attr)}'
                    else:
                        attributes_dict['variables'][attribute_v] = f' ~ {attribute_v}<{get_type_name(attr)}>'

        for attribute in dir(cls):
            if not attribute.startswith('__'):
                attr = getattr(cls, attribute)
                if callable(attr):
                    if verbose:
                        attributes_dict['methods'][attribute] = f' ~ {attribute}{get_method_info(attr)}'
                    else:
                        attributes_dict['methods'][attribute] = f' ~ {attribute}'
        return attributes_dict
    except Exception as e:
        print(ansi_color_str(f"Error occurred while getting class attributes: {e}", fg='red'))
        return {'variables': {}, 'methods': {}}

def get_class_attributes_color(cls, verbose=True):
    attributes_dict = {'variables': {}, 'methods': {}}

    try:
        for attribute_v in vars(cls):
            if not attribute_v.startswith('__') and 'stat' not in attribute_v:
                attr = getattr(cls, attribute_v)
                if not callable(attr):
                    if verbose:
                        attr_info = get_variable_info(attr)
                        attributes_dict['variables'][
                            attribute_v] = f' ~ {ansi_color_str(attribute_v, fg="green")} -> {ansi_color_str(attr_info, fg="cyan")}'
                    else:
                        attributes_dict['variables'][attribute_v] = f' ~ {ansi_color_str(attribute_v, fg="green")}'

        for attribute in dir(cls):
            if not attribute.startswith('__'):
                attr = getattr(cls, attribute)
                if callable(attr):
                    if verbose:
                        method_info = get_method_info(attr)
                        attributes_dict['methods'][
                            attribute] = f' ~ {ansi_color_str(attribute, fg="blue")} -> {ansi_color_str(method_info, fg="cyan")}'
                    else:
                        attributes_dict['methods'][attribute] = f' ~ {ansi_color_str(attribute, fg="blue")}'

        return attributes_dict
    except Exception as e:
        print(ansi_color_str(f"Error occurred while getting class attributes: {e}", fg='red'))
        return {'variables': {}, 'methods': {}}

def get_thread_info(message='', use_color=True):
    current_thread = threading.current_thread()
    current_thread_name = current_thread.name
    current_thread_id = current_thread.ident
    current_thread_alive = current_thread.is_alive()

    # Construct the colored thread info
    thread_info = ansi_color_str(f'thread:{current_thread_name}::{current_thread_id}::{current_thread_alive}',fg='yellow') if use_color else f'thread:{current_thread_name}::{current_thread_id}::{current_thread_alive}'
    formatted_message = f'{thread_info}:{message}'

    return formatted_message

def get_prev_caller_info(message='', steps=2, use_color=True):
    """
    Retrieve information about a previous caller in the call stack.

    Args:
        message (str): Additional message to include in the output.
        steps (int): How many steps back in the call stack to look.
        use_color (bool): Whether to include ANSI coloring for links.

    Returns:
        str: Formatted caller information string.
    """
    # Get the current frame
    frame = inspect.currentframe()

    try:
        # Move back `steps` frames in the call stack
        for _ in range(steps):
            if frame is None or frame.f_back is None:
                break
            frame = frame.f_back

        if frame is None:
            return "Frame not found"

        # Retrieve information about the selected frame
        frame_info = inspect.getframeinfo(frame)
        filename_with_path = frame_info.filename
        filename = os.path.basename(filename_with_path)
        linenumber = frame_info.lineno
        function = frame_info.function

        # Format the output string
        caller_link = (
            ansi_link_str(f"file:///{filename_with_path}", f"{filename}::{linenumber}::{function}")
            if use_color
            else f"{filename}::{linenumber}::{function}"
        )
        return f"{caller_link}: {message}"

    finally:
        # Clean up to prevent reference cycles
        del frame

def get_caller_info(message=''):
    # Get the current frame
    curr_frame = inspect.currentframe()
    # Get the caller's frame
    caller_frame = curr_frame.f_back
    # Retrieve the information about the caller's frame
    frame_info = inspect.getframeinfo(caller_frame)
    # Get the file name where the function was called
    filename_with_path = frame_info.filename
    # Extract only the file name
    filename = os.path.basename(filename_with_path)
    # Get the line number in the file where the function was called
    linenumber = frame_info.lineno
    # get the function name
    function = frame_info.function
    # Format the string to include the passed message
    caller_link = ansi_link_str(f"file:///{filename_with_path}", f"{filename}::{linenumber}::{function}") #if use_color else f"{filename}::{linenumber}::{function}"
    # caller_link = f"{filename}::{linenumber}::{function}"
    info_str = f"{caller_link}: {message}"  # file://
    # Clean up to prevent reference cycles
    del curr_frame
    del caller_frame
    return info_str

def get_call_stack_info(message='', steps=2, use_color=True):
    """
    Get a snapshot of the call stack around a specific frame depth.
    
    Args:
        message (str): Optional message to append.
        steps (int): The central frame index (default 2 = caller of caller).
        use_color (bool): Whether to use ANSI links.
    
    Returns:
        str: Multi-line string showing frames at steps-1, steps, steps+1.
    """
    stack_info = []
    frame = inspect.currentframe()

    try:
        # Walk up to steps+1 to get previous, current, and next frames
        frames = []
        for _ in range(steps + 2):  # +2 to make sure we can access steps+1
            if frame is None:
                break
            frames.append(frame)
            frame = frame.f_back

        def format_frame(f):
            if f is None:
                return " ~ | [Frame missing]"
            info = inspect.getframeinfo(f)
            filename = os.path.basename(info.filename)
            link = (
                ansi_link_str(f"file:///{info.filename}", f"{filename}::{info.lineno}::{info.function}")
                if use_color else f"{filename}::{info.lineno}::{info.function}"
            )
            return f" ~ | {link}"

        for offset in (-1, 0, 1):
            idx = steps + offset
            label = ["Prev", "Current", "Next"][offset + 1]
            stack_info.append(f"{label}: {format_frame(frames[idx] if idx < len(frames) else None)}")

        if message:
            stack_info.append(f" ~ | Msg: {message}")

        return "\n".join(stack_info)

    finally:
        del frame


def get_call_context_info(message='', steps=3, use_color=True):
    # Get the current frame
    curr_frame = inspect.currentframe()
    # Get the caller's frame by moving up 'steps' steps in the call stack
    caller_frame = curr_frame
    for _ in range(steps):
        caller_frame = caller_frame.f_back
        if caller_frame is None:
            break

    # Retrieve the information about the caller's frame
    frame_info = inspect.getframeinfo(caller_frame)
    # Get the file name where the function was called
    filename_with_path = frame_info.filename
    # Extract only the file name
    filename = os.path.basename(filename_with_path)
    # Get the line number in the file where the function was called
    linenumber = frame_info.lineno
    # Get the function name
    function = frame_info.function
    # Format the string to include the passed message
    caller_link = f"{filename}::{linenumber}::{function}"
    if use_color:
        caller_link = ansi_link_str(f"file:///{filename_with_path}", caller_link, link_color="bright_magenta")
    
    # Construct the hierarchical profile with indents
    indent_str = ' ' * 4  # Adjust the indentation level as needed
    hierarchical_profile = ''
    for i in range(steps):
        hierarchical_profile += f"{indent_str * i}{format_arrow('T', 'double', 3)} "
        hierarchical_profile += f"{caller_link}"
        if i != steps - 1:
            hierarchical_profile += '\n'

    info_str = f"{hierarchical_profile}: {message}"
    # Clean up to prevent reference cycles
    del curr_frame
    del caller_frame
    return info_str
