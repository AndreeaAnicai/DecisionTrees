To run the decision tree training and evaluation on a new dataset,
we have created a run script that executes training, evaluation and
cross validation with and without pruning and outputs a report to a file
and saves plots of an exemplary tree before and after pruning to your current
directory.

This can be run from the commandline as such:


python3 run.py <dataset you wish to use> <output file to save the report>


Please note that the entire procedure runs for a while (2-3 mins).
Afterwards you will find the results of the training and evaluation in
the output file specified.

Results include:
- Exemplary results without pruning (20% validation, 20% testing)
- Exemplary results with pruning (20% validation, 20% testing)
- Results of 10-fold cross-validation without pruning
- Results of 10-fold cross-validation with pruning
