urooj@hpcnexus-gpu-srv03 ~/test-adcewatt/test-mxm$
            ./run_basepower_adcewatt_var_std.py  // measure just baseline power
            
            ./run_appenergy_smxm_adcewatt.py  // simple without var,std
            
            ./run_appenergy_smxm_adcewatt_stdvar.py  // measure app energy wit std,var
            
run the command for application consumption measurement like this  --->> ./run_appenergy_smxm_adcewatt_stdvar.py --app-type serial     


================================= monitoring with pidstat (CPU-MEM only) =====================================================

also in this folder another file which is used to measure the cpu_mem use wile the script (./run_basepower_adcewatt_var_std.py) runs using comad pidstat:


just run this (CPU only) when just base script is running				./run_cpuidl_with_monitoring.sh
just run this (CPU only) when the application is running 				./run_cpu_with_monitoring_app.sh




================================= monitoring with pidstat (GPU only) =====================================================

just run this (CPU+gpu only) when the application is running 				./run_gpu_with_monitoring_app.sh
but currently this script is not working bcz the nvidia -smi comand is not working




================================= Bind my program on socket 1 cores =====================================================
Option 1 (recommended): bind to socket 1 using numactl

Change ONLY this line in your script (run_cpu_with_monitoring_app.sh)

Current line:
python3 run_appenergy_smxm_adcewatt_stdvar.py &

Replace it with:
numactl --cpunodebind=1 --membind=1 python3 run_appenergy_smxm_adcewatt_stdvar.py &

Option 2: bind to socket 1 using taskset (CPU-only)

Socket 1 CPUs are odd numbers, so:
taskset -c 1-127:2 python3 run_appenergy_smxm_adcewatt_stdvar.py &

How to verify it is really bound to socket 1

While the program is running:
numactl --show -p $SCRIPT_PID

**************** i have used the option 1 for my file:


========   also i binded the cores of my socket. in my case i have two options logical and physical cores as well.=========================
Option A: use all physical cores on socket 1 (no SMT noise)
Example (first 32 physical cores of socket 1):

numactl --physcpubind=1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97,101,105,109,113,117,121,125 \
        --membind=1 python3 run_appenergy_smxm_adcewatt_stdvar.py &
 
 
        
Option B: use all logical CPUs on socket 1

If your code benefits from SMT: 


       
numactl --physcpubind=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127 \
        --membind=1 python3 run_appenergy_smxm_adcewatt_stdvar.py &


**************** again i have used the option 1 for my file:

for socket 0:::

Socket 0 (physical cores only)

numactl --physcpubind=0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124 \
        --membind=0 python3 run_appenergy_smxm_adcewatt_stdvar.py &

======================================================= run_cpu_with_monitoring_app_bind.sh=====================================
new binded script and just run the file ./run_cpu_with_monitoring_app_bind.sh


