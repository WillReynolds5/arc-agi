1. REPO SETUP: clone note book and organize into repo

2. BUILD PROMPT AND RUN FIRST SAMPLE: in a scratch file, compute inference on some problems, the prompt should take in the examples and find the patterns and attempt to solve the problem

3. BUILD THE EVAL METRIC: find a test case where the llm is unable to solve the problem and build the module to do the following:
    a. run N prompt inference with the solve prompt to attempt the problem
    b. compute scores for all observations with custom eval metric "mean Intersection over Union (IoU)"
    c. analyze the scores distributions
    d. calculate average score, and return all observations above the average

4. BUILD THE LOOP: build the data generation script to iterate through the dataset and compute inference. The loop should work live this:
    a. run the eval loop over the entire dataset.
    b. add all training data to a dataset checkpoint

5. TRAIN THE MODEL: using the new training data, kick off a training run, perhaps lora to begin with an intelligent / large quantized model (gemma-27b)

6. RUN SYSTEM
    a. run the loop
    b. train model
    c. repeat steps

NOTES:
1. Ideally this runs completely autonomously in a h100 instance. 
