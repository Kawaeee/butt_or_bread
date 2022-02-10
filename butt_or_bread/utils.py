import psutil


def health_check():
    """Check CPU/Memory/Disk usage of deployed machine"""

    cpu_percent = psutil.cpu_percent(0.15)
    total_memory = psutil.virtual_memory().total / float(1 << 30)
    used_memory = psutil.virtual_memory().used / float(1 << 30)
    total_disk = psutil.disk_usage("/").total / float(1 << 30)
    used_disk = psutil.disk_usage("/").used / float(1 << 30)

    cpu_usage = f"CPU Usage: {cpu_percent:.2f}%"
    memory_usage = f"Memory usage: {used_memory:,.2f}G/{total_memory:,.2f}G"
    disk_usage = f"Disk usage: {used_disk:,.2f}G/{total_disk:,.2f}G"

    return " | ".join([cpu_usage, memory_usage, disk_usage])
