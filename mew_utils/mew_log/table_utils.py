from tabulate import tabulate

from .ansi_utils import ansi_color_str
from .attr_utils import get_class_attributes


def transpose_dict(d):
    return {k: [dic[k] for dic in d.values()] for k in d.keys()}


def format_data_as_table(data, headers=None, tablefmt='grid', **kwargs):
    if kwargs.get('use_color', False):
        # Apply color to headers if applicable
        if headers:
            headers = [ansi_color_str(header, fg="yellow") for header in headers]


    return tabulate(data, headers=headers, tablefmt=tablefmt)

def format_property(name, data, headers=None, tablefmt='simple_grid', use_color=False):
    colored_name = ansi_color_str(f"===[{name}]<{type(data).__name__}>===", fg="yellow") if use_color else f"===[{name}]<{type(data).__name__}>==="
    return f'{colored_name}\n{format_table(data, headers, tablefmt, use_color=use_color)}'



# note: pretty sure transpose doesn't work correctly
def format_table(data, headers=None, tablefmt='simple_grid', **kwargs):
    if not data:
        return ""
    if isinstance(data, dict):
        return format_dict_as_table(data, headers, tablefmt, **kwargs)
    elif isinstance(data, (list, set)):
        # transpose = list(zip(*data))
        return format_list_as_table(data, headers, tablefmt, **kwargs)
    elif hasattr(data, '__dict__'):
        return format_obj_as_table(data, headers, tablefmt, **kwargs)
    else:
        use_color = kwargs.get('use_color', False)
        if use_color:
            return ansi_color_str(str(data), fg='bright_yellow')
        else:
            return str(data)
            


def format_list_as_table(data, headers=None, tablefmt='simple', **kwargs):
    if not headers:
        headers = ["Index", "Value"] if kwargs.get('show_indexes', False) else []

    if kwargs.get('show_indexes', False):
        out_dict = {i: item for i, item in enumerate(data)}
        return format_table(out_dict, headers, tablefmt, **kwargs)
    else:
        formatted_str = ansi_color_str(f" | len={len(data)} | ", fg='bright_cyan') if kwargs.get('use_color', False) else f" | len={len(data)} | "
        formatted_str += "["
        for i, item in enumerate(data):
            formatted_str += format_table(item, [], tablefmt, **kwargs)
            if i < len(data) - 1: formatted_str += kwargs.get('delimeter', ', ')
        formatted_str += "]"
        return formatted_str


def format_dict_as_table(data, headers=None, tablefmt='simple_grid', **kwargs):
    if  kwargs.get('transpose', False):
        if not headers:
            headers =list(data.keys())
        #Transpose the dictionary: each key-value pair becomes a column
        transposed_data = [list(value) for key, value in zip(list(data.keys()), zip(*data.items()))]
        #intended for values of the same lenth
        #transposed_data = [headers] + [list(row) for row in zip(list(data.values()))]
        return format_data_as_table(transposed_data, headers=headers, tablefmt=tablefmt)
    else:
        if not headers:
            headers = ["Key", "Value"]

        # Convert the dictionary into a list of lists
        table_data = [[key, format_table(value, [], 'simple',**kwargs)] for key, value in data.items() if value is not None]

        # Format the table
        return format_data_as_table(table_data, headers=headers, tablefmt=tablefmt)


def format_obj_as_table(data, headers=None, tablefmt='fancy_grid', **kwargs):
    verbose = kwargs.get('verbose', True)
    fancy = kwargs.get('fancy', True)
    use_color = kwargs.get('use_color', True)
    
    attributes_dict = get_class_attributes(data, verbose=verbose)
    class_name = data.__class__.__name__
    variables = [*attributes_dict['variables'].values()]
    methods = [*attributes_dict['methods'].values()]
    # Check if headers are provided, if not, construct them
    if not headers:
        headers = [class_name]
        if variables:
            headers.append('variables')
        if methods:
            headers.append('methods')

    # Initialize an empty list to store the formatted table data
    table_data = []

    # formatted_vars = []
    # for v in variables:
    #     format_v = format_arg(v, use_color=use_color, fancy=fancy)
    #     formatted_vars.append(format_v)

    # Add variables and methods data to the table data
    table_data.append([format_table(variables, ['variables'], 'simple', **kwargs) if variables else None,
                       format_table(methods, ['methods'], 'simple', **kwargs) if methods else None])

    table_data = list(zip(*table_data))

    # Return the formatted table
    return format_data_as_table(table_data, headers, tablefmt, **kwargs)


def print_table(msg, data, headers=None, tablefmt='fancy_grid'):
    print(f'==={msg}==>\n{format_table(data, headers, tablefmt)}')


def format_square_layout(data):
    if isinstance(data, (list, set)):
        result_count = len(data)
        rows, columns = calc_best_square(result_count)
        table_data = [data[i:i + columns] for i in range(0, len(data), columns)]
        return format_table(table_data)
    elif isinstance(data, dict):
        items = [(k, v) for k, v in data.items()]
        result_count = len(items)
        rows, columns = calc_best_square(result_count)
        table_data = [items[i:i + columns] for i in range(0, len(items), columns)]
        return format_table(table_data)
    elif hasattr(data, '__dict__'):
        items = [(k, v) for k, v in data.__dict__.items()]
        result_count = len(items)
        rows, columns = calc_best_square(result_count)
        table_data = [items[i:i + columns] for i in range(0, len(items), columns)]
        return format_table(table_data)
    else:
        return format_table(data)


def print_square_layout(msg, data):
    print(f'==={msg}==>\n{format_square_layout(data)}')


def print_as_square(strings):
    # Calculate the number of strings in the input list
    result_count = len(strings)

    # Calculate the best square layout dimensions based on the result count
    rows, columns = calc_best_square(result_count)

    # Create a grid with empty strings filled in for each cell
    grid = [[' ' for _ in range(columns)] for _ in range(rows)]

    # Iterate over the strings and populate the grid with them
    for i, string in enumerate(strings):
        # Calculate the row and column index for the current string
        row = i // columns
        col = i % columns
        # Ensure the row and column indices are within the valid range of the grid dimensions
        if row < rows and col < columns:
            # Place the string in the corresponding cell of the grid
            grid[row][col] = string

    # Determine the maximum width of each column in the grid
    max_widths = [max(len(cell) for cell in row) for row in grid]

    # Print the grid, ensuring each cell is left-aligned and padded to its maximum width
    for row in grid:
        print(' '.join(cell.ljust(width) for cell, width in zip(row, max_widths)))

