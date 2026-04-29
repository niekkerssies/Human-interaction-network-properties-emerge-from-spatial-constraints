# Human interaction network properties emerge from spatial constraints
This code accompanies the paper of the same name (forthcoming). It's a pipeline to calibrate agent-based models to mobility behavior, and validating the model to bluetooth proximity data using Approximate Bayesian Computation. The results show that several key observed network properties of the Copenhagen Social Network study dataset can be generated from spatial constraints alone, including clustering, shortest path lengths, and the heavy tailed distribution of interactions per dyad.

SCRIPTS

The folder contains 4 python files, which can simply be downloaded and run in order. The first file, "Preprocessing.py", assumes "bt_symmetric.csv" is in that same folder; this is the bluetooth proximity data from the Copenhagen Social Network, available at https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433. In this file, networks are constructed from the data. In the second file, "Models.py", we define two agent-based models, both of which have agents move around in a 2D space and creating a network link while within range. In the first model, movement is random; in the second, it is bound to daily schedules, concentrating movement on specific places and times. In the third file, "ABC runs.py", we use the pyabc package to implement a likelihood-free inference method, designed to test the fit of complex models to complex data. By default, this file runs the first of 4 options, but any combination of options can be picked by changing the list RUN_TESTS. In the fourth file, "Collecting results.py", run results are collected and visualized.

More detailed documentation is found in the files themselves.


HOW TO RUN

1. Download "bt_symettric.csv" from https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433

1. Download .py scripts into same folder

3. Install packages if necessary

4. Optional: set one ore more desired ABC Runs by updating RUN_TESTS in "3. ABC Runs.py"

5. Run "ABC runs.py", then "Collecting results.py"


This is a visualization of what the second of our models does:

<img width="510" height="714" alt="Final SM model GIF" src="https://github.com/user-attachments/assets/4fab3f7c-587f-48c0-b860-e113999537af" />

The results visualizations look like this:

<img width="1890" height="1181" alt="Figure_3_but_really-1" src="https://github.com/user-attachments/assets/de26218a-fcd2-4380-abc8-8e632adc5f32" />










