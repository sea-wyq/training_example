#!/bin/bash

set -x

ROLE=$1
JOBNAME=$2
service_ip="112.95.163.96"   
service_port="28080" 

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$current_time] Starting script, JOBNAME: $JOBNAME"

if [ "$ROLE" == "master" ]; then
    pod_ip=$(printenv POD_IP)
    if [ -z "$pod_ip" ]; then
        echo "[$current_time] POD_IP is nil."
    fi
    response_code=$(curl -o /dev/null -s -X "POST" "http://$service_ip:$service_port/trainngregistry/upload/$JOBNAME" --data "ip=$pod_ip" -w %{http_code})
    if [ "$response_code" -eq 200 ]; then
        echo "[$current_time] Master IP upload successful."
    else
        echo "[$current_time] Master IP upload failed with status code: $response_code"
    fi
elif [ "$ROLE" == "worker" ]; then
    for (( i=0; i<60; i++ )); do
        response=$(curl "http://$service_ip:$service_port/trainngregistry/download/$JOBNAME")
        echo "[$current_time] Attempt $((i+1)): $response"
        if [ $? -eq 0 ]; then
            ip=$(echo "$response" | grep -oP '(?<="ip":")[^"]+')
            if [ -n "$ip" ]; then
                echo "[$current_time] Worker request successful after attempt $((i+1))."
                echo $ip > $JOBNAME.env
                echo "[$current_time] VJ_MASTER_IP set to $ip"
                break
            else
                echo "[$current_time] Failed to parse IP from response."
            fi
        else
            echo "[$current_time] Worker request failed, attempt $((i+1))"
        fi
        sleep 10 
    done
else
    echo "[$current_time] Invalid type parameter."
    exit 1
fi

