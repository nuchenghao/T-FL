num=$1
start=$2
> pids.txt
pid_file="pids.txt"


for ((i=0;i<num;i++)); do
   python3 client.py --name client$((i+start)) &
   echo $! >> $pid_file
done

wait