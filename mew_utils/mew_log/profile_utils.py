import time

from .ansi_utils import ansi_color_str
from .attr_utils import get_prev_caller_info

g_bot_ai = None

def get_last_step_time():
    if g_bot_ai:
        min_step, avg_step, max_step, last_step = g_bot_ai.step_time
        return last_step

# helper funcs
class ProfileSection:
    Registry = {}  # Class Variable

    def __init__(self, newMsg=None, enable=True, do_print=False, bot_ai=None):
        #global g_bot_ai

        #if not self.ai and bot_ai:
        self.ai = bot_ai

        self.msg = newMsg if newMsg else get_prev_caller_info()
        self.enable = enable
        self.startTime = 0.0
        self.endTime = 0.0
        self.elapsedTime = 0.0
        self.totalTime = 0.0
        self.avgTime = 0.0
        self.numCalls = 0
        self.do_print = do_print
        if self.msg in ProfileSection.Registry:
            p = ProfileSection.Registry[self.msg]
            self.__dict__ = p.__dict__
            self.numCalls += 1
        else:
            self.numCalls += 1

        if (self.enable):
            self.start()
    def __del__(self):
        if (self.enable):
            self.stop()

    def __enter__(self):
        # if self.enable:
        self.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # if self.enable:
        self.stop()

    def start(self):
        self.startTime = time.time() if not self.ai else self.ai.time
        # if self.do_print : log_info(str(self))

    def stop(self):
        self.endTime = time.time() if not self.ai else self.ai.time
        self.updateStats()
        stopMsg = str(self)

        p = create_or_get_profile(self.msg)
        p.__dict__.update(self.__dict__)
        if self.do_print: log_info(str(p))

        # step_time = get_last_step_time()
        # if step_time and step_time > 99:
        #     if self.elapsedTime * 1000 > 0.0:
        #         print(f"step_time: {step_time}")
        #         print(str(p))
            #g_bot_ai.client.chat_send(str(p))

        # st.toast(stopMsg)
        return stopMsg

    def updateStats(self):
        self.elapsedTime = self.endTime - self.startTime
        self.totalTime += self.elapsedTime
        self.avgTime = self.totalTime / float(self.numCalls)

    def __str__(self):
        msg_color = ansi_color_str(self.msg, fg='green')  # Green color for message
        elapsed_time_str = make_time_str('elapsed', self.elapsedTime)
        total_time_str = make_time_str('total', self.totalTime)
        avg_time_str = make_time_str('avg', self.avgTime)
        calls_info = f'calls={self.numCalls}'

        elapsed_color = ansi_color_str(elapsed_time_str, fg='cyan')  # Cyan color for elapsed time
        total_color = ansi_color_str(total_time_str, fg='cyan')  # Cyan color for total time
        avg_color = ansi_color_str(avg_time_str, fg='cyan')  # Cyan color for average time
        calls_info_color = ansi_color_str(calls_info, fg='cyan')  # Cyan color for calls information

        return f"{msg_color} ~ Elapsed Time: {elapsed_color} | Total Time: {total_color} | Avg Time: {avg_color} | {calls_info_color}"
    def to_str(self):
        """Generate a plain-text string representation."""
        elapsed_time_str = make_time_str('elapsed', self.elapsedTime)
        total_time_str = make_time_str('total', self.totalTime)
        avg_time_str = make_time_str('avg', self.avgTime)
        calls_info = f'calls={self.numCalls}'

        # if self.ai:
        #     elapsed_game_str = make_time_str('elapsed_game', self.elapsedTime)
        #     total_game_str = make_time_str('total_game', self.totalTime)
        #     avg_game_str = make_time_str('avg_game', self.avgTime)
        # else:
        #     elapsed_game_str = total_game_str = avg_game_str = "N/A"

        return (
            f"{self.msg} ~ "
            f"Elapsed Time: {elapsed_time_str} | Total Time: {total_time_str} | Avg Time: {avg_time_str} | "
            #f"Elapsed Game: {elapsed_game_str} | Total Game: {total_game_str} | Avg Game: {avg_game_str} | "
            f"{calls_info}"
        )

def make_time_str(msg, value):
    # do something fancy
    value, time_unit = (value / 60, 'min') if value >= 60 else (value * 1000, 'ms') if value < 0.101 else (value, 's')
    return f"{msg}={int(value) if value % 1 == 0 else value:.2f} {time_unit}"


def create_or_get_profile(key, enable=False, do_print=False):
    if key not in ProfileSection.Registry:
        ProfileSection.Registry[key] = ProfileSection(key, enable, do_print)
    return ProfileSection.Registry[key]


def profile_start(msg, enable=True, do_print=False):
    p = create_or_get_profile(msg, enable, do_print)
    if not enable: p.start()


def profile_stop(msg):
    if key in ProfileSection.Registry:
        create_or_get_profile(msg).stop()


def get_profile_registry():
    return ProfileSection.Registry


def get_profile_reports():
    reports = [value for value in ProfileSection.Registry.values()]
    reports.sort(key=lambda x: (x.totalTime, x.avgTime), reverse=True)
    return reports

def log_profile_registry():
    formatted_output = format_profile_registry()
    print(formatted_output)
    return formatted_output

def format_profile_registry(min_avg_time=0.01):
    """Format the profile registry, categorizing functions correctly and ensuring no duplicates."""

    reports = get_profile_reports()

    # Sort reports by (total time, avg time), descending order
    reports.sort(key=lambda x: (x.totalTime, x.avgTime), reverse=True)

    out_str = []
    out_str.append(f"=== Profile Reports (Filtered: avg_time >= {min_avg_time:.3f}s) ===\n")

    # Containers for different report categories
    significant_reports = []
    low_impact_reports = []
    zero_time_reports = []

    for p in reports:
        if p.avgTime >= min_avg_time:
            significant_reports.append(p)
        elif p.elapsedTime > 0 or p.totalTime > 0 or p.avgTime > 0:
            low_impact_reports.append(p)
        else:
            zero_time_reports.append(p)

    # Print Significant Reports
    if not significant_reports:
        out_str.append("No significant function times detected.\n")
    else:
        for profile in significant_reports:
            out_str.append(profile.to_str())
            out_str.append("-" * 80)

    # Print Low Impact Reports
    if low_impact_reports:
        out_str.append("\n=== Functions with Low Impact Detected ===\n")
        for profile in low_impact_reports:
            out_str.append(profile.to_str())

    # Print Zero Time Reports
    if zero_time_reports:
        out_str.append("\n=== Functions with Zero Time Detected ===\n")
        for profile in zero_time_reports:
            out_str.append(profile.to_str())

    return '\n'.join(out_str)


