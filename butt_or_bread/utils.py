import psutil


def health_check() -> str:
    """
    Check the CPU, memory, and disk usage of the deployed machine.

    Returns:
        A string with information about the current usage levels for each resource, formatted as follows:
        "CPU Usage: [percent]%" | "Memory usage: [used memory]GB/[total memory]GB" | "Disk usage: [used disk]GB/[total disk]GB"

    Uses the `psutil` module to obtain information about the system resources.
    """
    vm = psutil.virtual_memory()
    du = psutil.disk_usage("/")
    cpu_percent = psutil.cpu_percent(0.15)

    total_memory = vm.total / 1024**3
    used_memory = vm.used / 1024**3
    total_disk = du.total / 1024**3
    used_disk = du.used / 1024**3

    return f"CPU Usage: {cpu_percent:.2f}% | Memory usage: {used_memory:.2f}GB/{total_memory:.2f}GB | Disk usage: {used_disk:.2f}GB/{total_disk:.2f}GB"
