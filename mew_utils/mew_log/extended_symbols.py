import enum

class ExtendedSymbols:
    _symbol_map = {}
    class Symbol(enum.Enum):
        # Correctly differentiating between single and double line symbols
        TOP_RIGHT_L = '\u2510'        # ┐ Single line top right corner
        BOTTOM_RIGHT_L = '\u2518'     # ┘ Single line bottom right corner
        TOP_LEFT_L = '\u250C'         # ┌ Single line top left corner
        BOTTOM_LEFT_L = '\u2514'      # └ Single line bottom left corner
        RIGHT_T = '\u2524'            # ┤ Single line right T
        LEFT_T = '\u251C'             # ├ Single line left T
        TOP_T = '\u252C'              # ┬ Single line top T
        BOTTOM_T = '\u2534'           # ┴ Single line bottom T
        CROSS = '\u253C'              # ┼ Single line cross
        SINGLE_HORIZONTAL = '\u2500'  # ─ Single horizontal line
        SINGLE_VERTICAL = '\u2502'    # │ Single vertical line
        
        # Double line versions
        DOUBLE_TOP_RIGHT_L = '\u2557'  # ╗ Double line top right corner
        DOUBLE_BOTTOM_RIGHT_L = '\u255D'  # ╝ Double line bottom right corner
        DOUBLE_TOP_LEFT_L = '\u2554'   # ╔ Double line top left corner
        DOUBLE_BOTTOM_LEFT_L = '\u255A'  # ╚ Double line bottom left corner
        DOUBLE_RIGHT_T = '\u2563'      # ╣ Double line right T
        DOUBLE_LEFT_T = '\u2560'       # ╠ Double line left T
        DOUBLE_TOP_T = '\u2566'        # ╦ Double line top T
        DOUBLE_BOTTOM_T = '\u2569'     # ╩ Double line bottom T
        DOUBLE_CROSS = '\u256C'        # ╬ Double line cross
        DOUBLE_HORIZONTAL = '\u2550'   # ═ Double horizontal line
        DOUBLE_VERTICAL = '\u2551'     # ║ Double vertical line

        SOLID_BLOCK = '\u2588'         # █ Solid block
        TOP_HALF_BLOCK = '\u2580'      # ▀ Top half block
        BOTTOM_HALF_BLOCK = '\u2584'   # ▄ Bottom half block
        SHADED_BLOCK = '\u2592'        # ▒ Shaded block

    def __getattr__(self, name):
        if name in self._symbol_map:
            return self._symbol_map[name]
        try:
            value = self.Symbol[name].value
            self._symbol_map[name] = value
            return value
        except KeyError:
            raise AttributeError(f"No such symbol: {name}")

# Make ExtendedSymbols an instance so it can be used directly
ExtendedSymbols = ExtendedSymbols()

def format_header(header_msg, width=80, line_type='double'):
    

    formatted_header_msg = "【"+ header_msg + "】"

    header_len = len(formatted_header_msg)
    
    free_width = width - header_len

    header_line_half_width = free_width / 2

    header_segment_str = ExtendedSymbols.SINGLE_HORIZONTAL * header_line_segment if line_type == 'single' else ExtendedSymbols.DOUBLE_HORIZONTAL
    
    header_line_str =f'''{header_segment_str}{formatted_header_msg}{header_segment_str}'''
    return header_line_str

def format_box(msg=None, width=None, height=None, title=None, line_type='double'):
    # Use extended symbols correctly with Unicode
    line_char = ExtendedSymbols.SINGLE_HORIZONTAL if line_type == 'single' else ExtendedSymbols.DOUBLE_HORIZONTAL
    corner_tl = ExtendedSymbols.TOP_LEFT_L if line_type == 'single' else ExtendedSymbols.DOUBLE_TOP_LEFT_L
    corner_tr = ExtendedSymbols.TOP_RIGHT_L if line_type == 'single' else ExtendedSymbols.DOUBLE_TOP_RIGHT_L
    corner_bl = ExtendedSymbols.BOTTOM_LEFT_L if line_type == 'single' else ExtendedSymbols.DOUBLE_BOTTOM_LEFT_L
    corner_br = ExtendedSymbols.BOTTOM_RIGHT_L if line_type == 'single' else ExtendedSymbols.DOUBLE_BOTTOM_RIGHT_L
    vertical_line = ExtendedSymbols.SINGLE_VERTICAL if line_type == 'single' else ExtendedSymbols.DOUBLE_VERTICAL

    # Determine width based on message length if not specified
    if msg and (width is None or height is None):
        width = max(len(msg), width if width else 0) + 2  # Add padding
        height = 3  # Minimum box height with top, middle, and bottom

    # Use default dimensions if none provided
    if width is None:
        width = 10  # Default width
    if height is None:
        height = 3  # Default minimum height

    # Draw top border
    top_border = f"{corner_tl}{line_char * width}{corner_tr}"
    if title:
        title_str = f" {title} "
        title_length = len(title_str)
        pre_title_length = (width - title_length) // 2
        post_title_length = width - title_length - pre_title_length
        top_border = f"{corner_tl}{line_char * pre_title_length}{title_str}{line_char * post_title_length}{corner_tr}"

    # Prepare message line or blank lines
    if msg:
        message_line = f"{vertical_line} {msg} {vertical_line}"
        if len(msg) < width - 2:
            msg = msg + ' ' * (width - 2 - len(msg))
        message_line = f"{vertical_line}{msg}{vertical_line}"
    else:
        message_line = f"{vertical_line}{' ' * (width)}{vertical_line}"

    # Draw middle lines
    middle_lines = (f"{vertical_line}{' ' * width}{vertical_line}\n") * (height - 2)
    middle_lines = middle_lines[:-1]  # Remove the last newline for proper formatting

    # Draw bottom border
    bottom_border = f"{corner_bl}{line_char * width}{corner_br}"

    # Combine all parts
    box = f"{top_border}\n{message_line}\n{middle_lines}\n{bottom_border}"
    return box

def draw_box(msg=None, width=None, height=None, title=None, line_type='double'):
    box = format_box(msg, width, height, title, line_type)
    print(box)

def format_arrow(root_symbol=None, line_type='double', length=3):
    """
    Draw an arrow with a specified root symbol, line type, and length.
    """
    symbol_map = {
        'L': {
            'single': ExtendedSymbols.BOTTOM_LEFT_L,
            'double': ExtendedSymbols.DOUBLE_BOTTOM_LEFT_L,
        },
        'T': {
            'single': ExtendedSymbols.LEFT_T,
            'double': ExtendedSymbols.DOUBLE_LEFT_T
        },
        '+': {
            'single': ExtendedSymbols.CROSS,
            'double': ExtendedSymbols.DOUBLE_CROSS 
        }
    }

    # Default to a single horizontal line if undefined symbol or line type
    chosen_symbol = symbol_map.get(root_symbol, {}).get(line_type, '')
    line_char = ExtendedSymbols.DOUBLE_HORIZONTAL if line_type == 'double' else ExtendedSymbols.SINGLE_HORIZONTAL

    # Construct the arrow string
    return chosen_symbol + line_char * length + ">"

def draw_arrow(root_symbol=None, line_type='double', length=3):
    format_arrow_str = format_arrow(root_symbol, line_type, length)
    print(format_arrow_str)


def calc_best_square(result_count):
    # Calculate the closest square layout dimensions for result count
    square_side = int(result_count ** 0.5)

    # Check if the square is perfect
    if square_side ** 2 == result_count:
        return square_side, square_side


    # If not, find the closest square dimensions
    columns = square_side
    while columns > 1:
        if result_count % columns < result_count / columns:
            rows = result_count // columns
            return rows, columns
        columns -= 1

    # If unable to find exact square or rectangle, default to a single column
    return result_count, 1

def format_square_layout(data):
    if isinstance(data, (list, set)):
        result_count = len(data)
        rows, columns = calc_best_square(result_count)
        formatted_data = []
        for d in data:
            try:
                formatted_data.append(format_box(msg=d))
            except Exception as e:
                print("Error formatting data: {}", e)
        table_data = [formatted_data[i:i + columns] for i in range(0, len(formatted_data), columns)]
        return '\n'.join(['\n'.join(row) for row in table_data])
    elif isinstance(data, dict):
        items = [(k, v) for k, v in data.items()]
        result_count = len(items)
        rows, columns = calc_best_square(result_count)
        formatted_data = []
        for k, v in items:
            try:
                formatted_data.append(format_box(msg=f"{k}: {v}"))
            except Exception as e:
                print("Error formatting data: {}", e)
        table_data = [formatted_data[i:i + columns] for i in range(0, len(formatted_data), columns)]
        return '\n'.join(['\n'.join(row) for row in table_data])
    elif hasattr(data, '__dict__'):
        items = [(k, v) for k, v in data.__dict__.items()]
        result_count = len(items)
        rows, columns = calc_best_square(result_count)
        formatted_data = []
        for k, v in items:
            try:
                formatted_data.append(format_box(msg=f"{k}: {v}"))
            except Exception as e:
                print("Error formatting data: {}", e)
        table_data = [formatted_data[i:i + columns] for i in range(0, len(formatted_data), columns)]
        return '\n'.join(['\n'.join(row) for row in table_data])
    else:
        try:
            return format_box(data)
        except Exception as e:
            print("Error formatting data: {}", e)

