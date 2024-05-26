##!/bin/bash
#
## 水平分割窗格
#tmux split-window -h
## 垂直分割前两个窗格
#tmux split-window -v
#tmux select-pane -t 0
#tmux split-window -v
## 切换到第二个窗格并垂直分割
#tmux select-pane -t 2
#tmux split-window -v
#
## 等待一小段时间以确保pane已经创建
#sleep 2
#
## 在每个pane中激活conda环境并执行Python脚本
## 每个client指定一个核心
#for i in {0..4}
#do
#
#   tmux send-keys -t $i "conda activate nch && taskset -c $i python3 client.py --name client$i" C-m
#done

#!/bin/bash

num=$1
start=$2
> pids.txt
pid_file="pids.txt"


for ((i=0;i<num;i++)); do
   taskset -c $i python3 client.py --name client$((i+start)) &
   echo $! >> $pid_file
done

wait