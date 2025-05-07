from ..mew_config.mew_config import MewConfig

from ..mew_log.log_utils import init_log

# general

class LogConfig(MewConfig):
    def add_default_config_values(self):
        self.add_to_config("enable_log", True)
        self.add_to_config("log_to_file", False)
        self.add_to_config("clear_file_first", False)
        self.add_to_config("log_filename", 'log')
        self.add_to_config("log_filedir", 'data/logs/')
        self.add_to_config("log_filepath", f"{self['log_filedir']}{self['log_filename']}.txt")
        # Call the superclass method to include default values from the parent class
        super().add_default_config_values()

    def apply_config(self):
        init_log(self)





# initialize all

g_logConfig = LogConfig('log', preload=True)
