#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if a file with this timestamp already exists (in case of multiple runs in the same second)
COUNTER=1
BASE_NAME="logs/activity_${TIMESTAMP}"

while [ -f "${BASE_NAME}_${COUNTER}.log" ]; do
    COUNTER=$((COUNTER + 1))
done

# Final log filename
LOGFILE="${BASE_NAME}_${COUNTER}.log"

echo "=========================================="
echo "Recording session to: $LOGFILE"
echo "=========================================="
echo ""

# Start recording
script -a "$LOGFILE"
