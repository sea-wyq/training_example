#!/bin/bash

jobname=$1
namespace=$2
suffix=$3
  
cat /etc/colossalai/hostfile | while read pod_name  
do
   dns_name="$pod_name.$jobname.$namespace.$suffix"
   pod_ip=$(host $dns_name | grep "has address" | sed 's/has address/-/g' |  awk '{print $3,$1}') 
   echo "$pod_ip $pod_name" >> /etc/hosts     
done    