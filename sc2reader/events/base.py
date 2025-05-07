# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals, division


def format_event(event):
    # Format timestamp
    time_sec = getattr(event, 'second', None)
    if time_sec is not None:
        m, s = divmod(int(time_sec), 60)
        time_str = f"{{{m}}}m{{{s}}}s"
    else:
        time_str = "{??}m{??}s"

    # Flatten dict to key=value string
    kv_list = [f"{k}={v}" for k, v in event.__dict__.items()]
    kv_str = ", ".join(kv_list)

    return f" ~ | second[{time_str}] ~ {event.__class__.__name__}: {kv_str}"



class Event(object):
    name = "Event"

    def __str__(self):
        return format_event(self)
