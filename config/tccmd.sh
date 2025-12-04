tc qdisc add dev eth0 root tbf rate 100mbit burst 100kbit latency 100ms

tc qdisc add dev eth0 ingress

tc filter add dev eth0 protocol ip parent root prio 1 u32 match ip dst 0.0.0.0/0 police rate 10mbit burst 100kbit drop flowid :1