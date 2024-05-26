pid_file="pids.txt"

while read pid; do
    kill $pid
    echo "已杀掉进程 $pid"
done < $pid_file