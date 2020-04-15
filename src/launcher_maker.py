import os
import sys

test_name = sys.argv[1]
print("Test:", test_name)
f = open("main_launcher.sh", "w")
f.write("#!/bin/bash\n")

f.write('#SBATCH --job-name="' + test_name + '"\n')

f.write('#SBATCH --qos=training\n')

f.write('#SBATCH --workdir=.\n')

f.write('#SBATCH --output=../logs/' + test_name + '/model.out\n')

f.write('#SBATCH --error=../logs/' + test_name + '/model.err\n')

f.write('#SBATCH --cpus-per-task=40\n')

f.write('#SBATCH --gres gpu:1\n')

f.write('#SBATCH --time=10:00:00\n')

f.write('module purge; module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML\n')

f.write('python ../logs/' + test_name + '/main.py ../architechtures/config.json\n')
f.close()

command = os.popen('mkdir ../logs/' + test_name)
print(command.read())
command.close()
print("Folder created.")
command = os.popen('cp * ../logs/' + test_name)
print(command.read())
command.close()
print("Structure copied.")
command = os.popen('sbatch main_launcher.sh')
print(command.read())
command.close()
print("REMEMBER THAT THE PASSED PARAMETER TO main_launcher.sh MUST BE THE SAME AS THE 'test_name' IN config.json")
