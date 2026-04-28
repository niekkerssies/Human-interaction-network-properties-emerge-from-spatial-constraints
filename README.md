# Human-interaction-network-properties-emerge-from-spatial-constraints
This code accompanies the paper of the same name (forthcoming). It's a pipeline to calibrate agent-based models to mobility behavior, and validating the model to bluetooth proximity data using Approximate Bayesian Computation.

SCRIPTS

The folder contains 4 python files, which can simply be downloaded and run in order (they need to run in order because the later scripts retrieve objects created by the earlier scripts). The first file, "1. Preprocessing.py", assumes "bt_symmetric.csv" is in that same folder; this is the bluetooth proximity data from the Copenhagen Social Network, available at https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433. In this file, networks are constructed from the data. In the second file, "2. Models.py", we define two agent-based models, both of which have agents move around in a 2D space and creating a network link while within range. In the first model, movement is random; in the second, it is bound to daily schedules, concentrating movement on specific places and times. In the third file, "3. ABC runs.py", we use the pyabc package to implement a likelihood-free inference method, designed to test the fit of complex models to complex data. By default, this file runs the first of 4 options, but any combination of options can be picked by changing the list RUN_TESTS. In the fourth file, "4. Collecting results.py", run results are collected and visualized.

More detailed documentation is found in the files themselves.


HOW TO RUN

1. Download files into same folder

2. Install packages if necessary

3. Optional: set one ore more desired ABC Runs by updating RUN_TESTS in "3. ABC Runs.py"

4. Run files in order


This is a visualization of what the second of our models does:

<img width="510" height="714" alt="SM GIF" src="https://github.com/user-attachments/assets/0c01afea-01de-4949-b13f-41a357951e0f" />

The results visualizations look like this:

[Figure_3_but_really.pdf](https://github.com/user-attachments/files/27165202/Figure_3_but_really.pdf)










