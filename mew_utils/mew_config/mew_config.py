import json

from ..mew_log.json_utils import MewJsonEncoder, prettify
from ..mew_file import MewFile
from ..mew_log.attr_utils import get_caller_info
from ..mew_log.log_utils import log_info


class MewConfig(MewFile):
    Registry = {}

    def __init__(self, config_id: str, preload=False):
        self.config_id = config_id
        self.json_data = {}
        file_name = f"config_{self.config_id}"
        file_type = "config"
        self.loaded = False
        super().__init__("data/config/" + file_name, file_type)

        if preload:
            self.load_config()
        MewConfig.Registry[self.config_id] = self

    def add_default_config_values(self):
        self.add_to_config("mew", '77')

    def __getitem__(self, key):
        return self.json_data.get(key, None)

    def get_type(self, key):
        return type(self.__getitem__(key))

    def __setitem__(self, key, value):
        self.json_data[key] = value

    def get(self, key, default_value = None):
        return self.json_data.get(key, default_value)

    def set(self, key, value):
        self.json_data[key] = value

    def add_to_config(self, key, value):
        self.json_data[key] = value

    def update_config(self, json_data):
        self.json_data.update(json_data)

    def clear_config(self):
        self.json_data.clear()

    def save_config(self, filename=""):
        if filename == "": filename = f"data/config/config_{self.config_id}.json"
        log_info(get_caller_info(f"\n===config_{self.config_id}.json save begin..."))
        self.ensure_directory_exists()
        # self.log_config()
        with open(filename, "w") as file:
            saved_json_str = json.dumps(self.json_data, cls=MewJsonEncoder, indent=4)
            log_info(f"\n===saved_json_str={saved_json_str}\n===type={type(saved_json_str)}")
            file.write(f'{saved_json_str}')
        log_info(get_caller_info(f"\n===config_{self.config_id}.json saved to path: {self.path}"))

    def load_config(self):
        if not self.loaded:
            log_info(get_caller_info(f'{self.path}'))
            self.ensure_directory_exists()
            if self.exists():
                log_info(f"\n==={self.path} found...loading...")
                with open(self.path, "r") as file:
                    loaded_json_data = json.load(file, object_hook=MewJsonEncoder.mew_decoder_hook)
                    self.json_data.update(loaded_json_data)
                    self.log_config()
            else:
                log_info(f"\n==={self.path} not found. Using default values.")
                self.add_default_config_values()
                self.save_config()
            # apply after loading
            self.apply_config()
            self.loaded = True

    def apply_config(self):
        # override in sub classes
        pass

    def log_config(self):
        print(str(self))

    def log_config_value(self, key):
        log_info(f" - {key.upper()}: {self.json_data.get(key)}")

    def __str__(self):
        out_str = f"===config_id: {self.config_id}"  # super().__str__()
        if self.json_data:
            for key in self.json_data.keys():
                out_str += f"\n ~ {key.upper()}: {self.json_data.get(key)}"
        return out_str


def save_all_configs():
    log_info(get_caller_info(f' begin!'))
    for config in MewConfig.Registry.values():
        config.save_config()


def load_all_configs():
    log_info(get_caller_info(f' begin!'))

    # config_list = list(MewConfig.Registry.values())
    # log_info(get_caller_info(f'config_list: {config_list}'))

    for config in MewConfig.Registry.values():
        config.load_config()

    return MewConfig.Registry


def load_config(config_id: str):
    #log_info(get_caller_info(f' begin!'))
    mew_config = MewConfig.Registry.get(config_id)
    if mew_config:
        mew_config.load_config()
        return mew_config


def get_mew_config(config_id: str):
    return load_config(config_id)


def get_mew_config_registry():
    return MewConfig.Registry
