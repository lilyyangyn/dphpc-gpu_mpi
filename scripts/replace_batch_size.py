import subprocess
if __name__ == '__main__':

    # change the definition of block size in io.cuh
    f = open("/home/xindzuo/dphpc-gpu_mpi/gpu_libs/gpu_mpi/io.cuh", "r+")
    line_list = f.readlines()
    word_list = line_list[78].split()
    word_list[2] = str(int(word_list[2]) * 2)
    new_line = word_list[0] + " " + word_list[1] + " " + word_list[2] + "\n"
    line_list[78] = new_line
    print(line_list[78])
    f.truncate(0)
    f.seek(0)
    f.writelines(line_list)
    f.close()

    # # change the places where blocksize is used in benchmark
    # f = open("/home/xindzuo/dphpc-gpu_mpi/toolchain_tests/basicIO/best_block_size.c", "r+")
    # line_list = f.readlines()
    # word_list = line_list[4].split()
    # word_list[2] = str(int(word_list[2]) * 2)
    # new_line = word_list[0] + " " + word_list[1] + " " + word_list[2] + "\n"
    # line_list[4] = new_line
    # print(line_list[4])
    # f.truncate(0)
    # f.seek(0)
    # f.writelines(line_list)
    # f.close()

    # # run the benchmark and output the result into a file
    # def run_command(cmd=None, out=None, cwd=None):
    #     rt = subprocess.call(cmd, shell = True, stdout = out, encoding="utf-8", cwd = cwd)
    # # recompile the project
    # run_command(cmd = "bash /home/xindzuo/dphpc-gpu_mpi/scripts/recompile.sh", cwd = "/home/xindzuo/dphpc-gpu_mpi/build")
    # # run_command("bash /home/xindzuo/dphpc-gpu_mpi/scripts/recompile.sh")

    # # run the benchmark
    # f = open("/home/xindzuo/dphpc-gpu_mpi/toolchain_tests/basicIO/result_of_best_block_size.txt", "r+")
    # cmd = []
    # cmd.append("export GPU_MPI_PROJECT=/home/xindzuo/dphpc-gpu_mpi")
    # cmd.append("export PATH=$PATH:/usr/local/cuda-11.3/bin/")
    # cmd.append("/home/xindzuo/miniconda3/bin/python3.7 /home/xindzuo/dphpc-gpu_mpi/build/scripts/gpumpicc.py /home/xindzuo/dphpc-gpu_mpi/toolchain_tests/basicIO/best_block_size.c")
    # run_command(cmd=cmd, cwd="/home/xindzuo/dphpc-gpu_mpi//toolchain_tests/basicIO")
    # cmd = "/home/xindzuo/dphpc-gpu_mpi/toolchain_tests/basicIO/a.out ---gpumpi -g 1 -b 1"
    # run_command(cmd=cmd, cwd="/home/xindzuo/dphpc-gpu_mpi//toolchain_tests/basicIO")
    # f.close()

