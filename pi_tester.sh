#!/bin/bash

log_file="$(date +"%Y-%m-%d")-temperature_log.txt"
sudo rm "CPU-top.txt"
sudo rm "CPU-mpstat.txt"
sudo rm "$log_file"

while true; do
    temperature=$(vcgencmd measure_temp | cut -d "=" -f2)
    voltage=$(vcgencmd measure_volts)
    arm_clock=$(vcgencmd measure_clock arm)
    gpu_memory=$(vcgencmd get_mem gpu)
    throttled=$(vcgencmd get_throttled)

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    log_entry="$timestamp - Temperature: $temperature, Voltage: $voltage, ARM Clock: $arm_clock, GPU Memory: $gpu_memory, Throttled: $throttled"

    echo "$log_entry" >> "$log_file"
    echo "System information logged."

#    echo q | htop | aha --black --line-fix > "../$timestamp-htop.html"
    echo "\n\n###### CPU STATS $timestamp ######\n\n" >> "CPU-top.txt"
    echo "\n\n###### CPU STATS $timestamp ######\n\n" >> "CPU-mpstat.txt"
    top -bcn1 -w100 >> "CPU-top.txt"
    mpstat -P ALL >> "CPU-mpstat.txt"
    # Log every 2 minutes
    sleep 120
done
