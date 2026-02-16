In this folder, one can find every important file for the project of Parallel and Grid Computing for the Master in Informatics and Computer Science of Gabriel S.J. Spadoni.

The files dqn.py, dqn_parallel.py and dqn_exploitation.py are the python files with respecively the Reinforcement Learning Implementation, the parallel reinforcement learning implementation and the exploitation.

The file PGC_report_Gabriel_Spadoni.pdf contains the report of the project.

The folder additional_files contains:
	- the read_csv_file.py python file used to read the csv files of the results and create the plots
	- the launcher.sh and the launcher2.sh files used to launch the jobs on the HPC

The folder results contains:
	- the checkpoint_*.pt files which are the weights of the non parallel reinforcement learning models, they can be used with the  dqn_exploitation.py file.
	- the Loss_Seq_*.png files are the plot created with the python file read_csv_file.py using the results from the files run_seq_cuda_*.csv.
	- the run_seq_cuda_*.csv files are the loss values for 100, 200, 500 and 1000 iterations of the dqn.py file
	- the checkpoint_dpp*.pt files are the weights of the parallel reinforcement models, they can be loaded and used with the dqn_exploitation.py file
	- the run_1_*.csv files are the loss values for iterations of the dqn_parallel.py file
	- the Loss_Para_*.png are the plot created with the python file read_csv_file.py using the results from the files run_1_*.csv.
