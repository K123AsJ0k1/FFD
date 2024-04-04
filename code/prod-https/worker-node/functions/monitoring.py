import time
import os
import psutil

from datetime import datetime
# Created and works
def get_system_resource_usage():
    net_io_counters = psutil.net_io_counters()
    system_resources = {
        'name': 'system',
        'date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'time': time.time(),
        'cpu-percent': psutil.cpu_percent(interval = 0.1),
        'ram-percent': psutil.virtual_memory().percent,
        'ram-total-bytes': psutil.virtual_memory().total,
        'ram-free-bytes': psutil.virtual_memory().free,
        'ram-used-bytes': psutil.virtual_memory().used,
        'disk-total-bytes': psutil.disk_usage('.').total,
        'disk-free-bytes': psutil.disk_usage('.').free,
        'disk-used-bytes': psutil.disk_usage('.').used,
        'network-sent-bytes': net_io_counters.bytes_sent,
        'network-received-bytes': net_io_counters.bytes_recv,
        'network-packets-sent': net_io_counters.packets_sent,
        'network-packets-received': net_io_counters.packets_recv,
        'network-packets-sending-errors': net_io_counters.errout,
        'network-packets-reciving-errors': net_io_counters.errin,
        'network-packets-outgoing-dropped': net_io_counters.dropout,
        'network-packets-incoming-dropped': net_io_counters.dropin
    }
    return system_resources
# Created and works
def get_server_resource_usage():
    this_process = psutil.Process(os.getpid())
    cpu_percent = this_process.cpu_percent(interval = 0.1)
    memory_info = this_process.memory_full_info()
    disk_info = this_process.io_counters()
    server_resources = {
        'name': 'server',
        'date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'time': time.time(),
        'cpu-percent': cpu_percent,
        'ram-resident-set-size-bytes': memory_info.rss,
        'ram-virtual-memory-size-bytes': memory_info.vms,
        'ram-shared-memory-bytes': memory_info.shared,
        'ram-code-segment-size-bytes': memory_info.text,
        'ram-data-segment-size-bytes': memory_info.data,
        'ram-library-size-bytes': memory_info.lib,
        'ram-dirty-pages-bytes': memory_info.dirty,
        'ram-unique-set-size-bytes': memory_info.uss,
        'disk-read-bytes': disk_info.read_bytes,
        'disk-write-bytes': disk_info.write_bytes,
        'disk-read-characters-bytes': disk_info.read_chars,
        'disk-write-characters-bytes': disk_info.write_chars
    }
    return server_resources  