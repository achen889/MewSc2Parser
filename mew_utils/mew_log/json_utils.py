import json
import types
from enum import Enum

from .attr_utils import is_of_type, get_caller_info
from .log_utils import log_info


def to_json_data(data):
    if isinstance(data, (str, bytes, bytearray)):
        json_data = json.loads(data)
        return json_data
    elif isinstance(data, (list, set, dict)):
        return data
    elif hasattr(data, '__dict__'):
        return data.__dict__
    elif hasattr(data, 'dict') and callable(getattr(json_data, 'dict')) and len(json_data.dict()) > 0:
        return data.dict()
    return str(data)


def is_json_serializable(data):
    try:
        json.dumps(data)
        return True
    except TypeError:
        return False


def prettify(data):
    if is_json_serializable(data):
        json_data = to_json_data(data)
        return json.dumps(json_data, indent=4)
    else:
        return str(data)


def prettify_property(name, data):
    return f'[{name}]<{type(data)}> ~ {prettify(data)}'


def is_dict_empty(my_dict):
    if is_of_type(my_dict, dict, raise_err=True):
        return len(my_dict.items()) == 0


def fetch_property(json_data, name, enable_log=False):
    out_fetched = None
    if not is_dict_empty(json_data):
        if name in json_data:
            out_fetched = json_data[name]
            if out_fetched:
                if enable_log: log_info(get_caller_info(prettify_property(name, out_fetched)))
    return out_fetched


class MewJsonParser:
    def __init__(self, json_str):
        self.json_data = to_json_data(json_str)
        keys = self._json_data.keys()

    def __getitem__(self, key):
        out_fetched = self._json_data.get(key, None)
        if out_fetched:
            if enable_log: log_info(get_caller_info(prettify_property(name, out_fetched)))
        return out_fetched


class MewJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        out_encoded = {}
        out_encoded['type'] = f'{type(obj)}'
        # out_encoded['__mew_json_encoded__']['name'] = obj.__name__
        if isinstance(obj, Enum):
            # Convert enum instance to its value
            out_encoded['data'] = obj.value
            return out_encoded
        elif isinstance(obj, set):
            out_encoded['data'] = list(obj)
            return out_encoded
        elif isinstance(obj, list):
            out_encoded['data'] = list(obj)
            return out_encoded
        elif isinstance(obj, dict):
            out_encoded['data'] = obj.__dict__
            return out_encoded
        elif callable(obj) and isinstance(obj, types.FunctionType):
            out_encoded['name'] = obj.__name__
            out_encoded['module'] = obj.__module__
            return out_encoded
        elif isinstance(obj, type):
            # return super().default(obj)
            out_encoded['name'] = obj.__name__
            out_encoded['module'] = obj.__module__
            if hasattr(obj, '__dict__'):
                out_encoded['data_type'] = 'dict'
                out_encoded['data'] = obj.__dict__
                return out_encoded
        elif hasattr(obj, '__dict__'):
            out_encoded['data_type'] = 'dict'
            out_encoded['data'] = obj.__dict__
            return out_encoded
        elif hasattr(obj, '__slots__'):
            slots_data = {slot: getattr(obj, slot) for slot in obj.__slots__}
            out_encoded['data_type'] = 'slots'
            out_encoded['data'] = slots_data
            return out_encoded
        else:
            # For other types, use the default serialization method
            return super().default(obj)

    @staticmethod
    def mew_decoder_hook(obj):
        if '__mew_json_encoded__' in obj:
            mew_data = obj['__mew_json_encoded__']
            if mew_data['type'] == 'set':
                return set(mew_data['data'])
            elif mew_data['type'] == 'list':
                return list(mew_data['data'])
            elif mew_data['type'] == 'dict':
                return mew_data['data']
            elif mew_data['type'] == types.FunctionType:
                recreated_func = getattr(__import__(mew_data["module"]), mew_data["name"])
                return recreated_func
            # else:
            #     recreated_class = getattr(__import__(mew_data["module"]), mew_data["name"])
            #     if mew_data['data_type'] == 'dict':
            #         return recreated_class(**mew_data['data'])
        return obj


class MewJsonEncoder2(json.JSONEncoder):
    def default(self, obj):
        out_encoded = {'__mew_json_encoded__': {}}
        out_encoded['__mew_json_encoded__']['type'] = f'{type(obj)}'
        # out_encoded['__mew_json_encoded__']['name'] = obj.__name__
        if isinstance(obj, Enum):
            # Convert enum instance to its value
            out_encoded['__mew_json_encoded__']['data'] = obj.value
            return out_encoded
        elif isinstance(obj, set):
            out_encoded['__mew_json_encoded__']['data'] = list(obj)
            return out_encoded
        elif isinstance(obj, list):
            out_encoded['__mew_json_encoded__']['data'] = list(obj)
            return out_encoded
        elif isinstance(obj, dict):
            out_encoded['__mew_json_encoded__']['data'] = obj.__dict__
            return out_encoded
        elif callable(obj) and isinstance(obj, types.FunctionType):
            out_encoded['__mew_json_encoded__']['name'] = obj.__name__
            out_encoded['__mew_json_encoded__']['module'] = obj.__module__
            return out_encoded
        elif isinstance(obj, type):
            return super().default(obj)
            # out_encoded['__mew_json_encoded__']['name'] = obj.__name__
            # out_encoded['__mew_json_encoded__']['module'] = obj.__module__
            # if hasattr(obj, '__dict__'):
            #     out_encoded['__mew_json_encoded__']['data_type'] = 'dict'
            #     out_encoded['__mew_json_encoded__']['data'] = obj.__dict__
            #     return out_encoded
        elif hasattr(obj, '__dict__'):
            out_encoded['__mew_json_encoded__']['data_type'] = 'dict'
            out_encoded['__mew_json_encoded__']['data'] = obj.__dict__
            return out_encoded
        elif hasattr(obj, '__slots__'):
            slots_data = {slot: getattr(obj, slot) for slot in obj.__slots__}
            out_encoded['__mew_json_encoded__']['data_type'] = 'slots'
            out_encoded['__mew_json_encoded__']['data'] = slots_data
            return out_encoded
        else:
            # For other types, use the default serialization method
            return super().default(obj)

    @staticmethod
    def mew_decoder_hook(obj):
        if '__mew_json_encoded__' in obj:
            mew_data = obj['__mew_json_encoded__']
            if mew_data['type'] == 'set':
                return set(mew_data['data'])
            elif mew_data['type'] == 'list':
                return list(mew_data['data'])
            elif mew_data['type'] == 'dict':
                return mew_data['data']
            elif mew_data['type'] == types.FunctionType:
                recreated_func = getattr(__import__(mew_data["module"]), mew_data["name"])
                return recreated_func
            # else:
            #     recreated_class = getattr(__import__(mew_data["module"]), mew_data["name"])
            #     if mew_data['data_type'] == 'dict':
            #         return recreated_class(**mew_data['data'])
        return obj
