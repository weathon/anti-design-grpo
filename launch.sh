#!/bin/bash
# Start (or reuse) the HPSv3 server, then launch training into the next logs/train_N.log.
cd /lambda/nfs/wash/grpo || exit 1
if ! curl -s --noproxy '*' -o /dev/null "http://127.0.0.1:5000/" --max-time 5 && ! { [ -f logs/hps_pid ] && kill -0 "$(cat logs/hps_pid)" 2>/dev/null; }; then
  nohup /home/ubuntu/.hpsv3/bin/python -u hpsv3_serve.py > logs/hpsv3.log 2>&1 &
  echo $! > logs/hps_pid
fi
# wait for the server to answer (any HTTP status counts; model load can take a few minutes)
for _ in $(seq 120); do
  code=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:5000/" --max-time 5)
  [ "$code" != "000" ] && break
  kill -0 "$(cat logs/hps_pid)" 2>/dev/null || { echo "HPSv3 server died"; tail -20 logs/hpsv3.log; exit 1; }
  sleep 5
done
[ "$code" = "000" ] && { echo "HPSv3 server not up after 10 min"; exit 1; }
n=$(( $(ls logs/train_*.log 2>/dev/null | sed 's/[^0-9]//g' | sort -n | tail -1) + 1 ))
nohup /home/ubuntu/miniconda3/envs/grpo-vllm/bin/python -u main.py > logs/train_$n.log 2>&1 &
echo $! > logs/pid
echo "hpsv3 pid $(cat logs/hps_pid), train pid $(cat logs/pid), log logs/train_$n.log"
