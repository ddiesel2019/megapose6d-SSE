# Diego Machado Diesel

# This aims at solving errors regarding firefox and geckodriver not in PATH.
# This command should be run from within the conda environment megapose-SSE

conda install -c conda-forge firefox geckodriver
which firefox
which geckodriver

echo 'export PATH="$HOME/miniconda3/envs/megapose-SSE/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc

conda activate megapose-SSE
