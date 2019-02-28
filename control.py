import os
for i in range(0,5):
    os.system("python cifar10_distribute_Train.py --job_name=worker --task_index=0")