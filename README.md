## APGP - The Asynchronous Parallel Genetic Programming 


Instead of model size, APGP employs the *evaluation time* --- the computational time required to evaluate a GP model on data --- as a measure of its complexity. This notion is based on the observation that a model made up of computationally expensive building blocks or having large structures takes a long time to be evaluated, and hence is computationally complex. Therefore, the evaluation time control discourages both the structural as well as functional complexity. 

APGP takes a simple view: induce a *race* among competing models that allows a model to join the breeding population as soon as it has finished evaluating. Hence, the faster models can (fitness permitting) join the breeding population before their slower counterparts and gain an evolutionary advantage. This advantage arises because the competing models compete in terms of not only their accuracy but also their evaluation times due to the race; this is quite unlike in standard GP where each evaluation (or a batch of evaluations, as in generational replacement) is allowed to finish before the next evaluation (or a batch of evaluations) can start. Note, however, that selection is solely based on accuracy; therefore, APGP facilitates a dynamic interplay between accuracy and simplicity. To induce this race, APGP evaluates multiple models simultaneously across multiple asynchronous threads. 


## The fixed length initialisation scheme (FLI)
The option to use a fixed length initialisation scheme (FLI) is available in the code.

Fixed-Length Initialisation (FLI)}  scheme that can be used to start the evolution with a size converged and functionally diverse population.  The results demonstrated that using FLI significantly improves the test fitness on a variety of GP techniques both with bloat-control and time-control.


# Derived Pulications:

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah.  *“Evolving Simple and Accurate Symbolic Regression Models via Asynchronous Parallel Computing"* In: Applied Soft Computing 104 (2021), p. 107198. ISSN: 1568-4946.
 URL: https://doi.org/10.1016/j.asoc.2021.107198

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah.  *“Time control or size control? reducing complexity and improving the accuracy of genetic programming models"*, In: European Conference on Genetic Programming, Springer, 2020, pp. 195–210. URL: https://doi.org/10.1007/978-3-030-44094-7_13

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Leveraging asynchronous parallel computing to produce simple genetic programming computational models",* In: Proceedings of the 35th Annual ACM Symposium on Applied Computing, SAC ’20, Association for Computing Machinery, NY, USA, 2020, p521–528. URL: https://doi.org/10.1145/3341105.3373921

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Feature Engineering for Enhanced Performance of Genetic Programming Models",*
In: GECCO '20 Companion, Genetic and Evolutionary Computation Conference Companion, July 2020. URL: https://doi.org/10.1145/3377929.3390078.
 
Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Improving the Generalisation of Genetic Programming Models with Evaluation Time and Asynchronous Parallel Computing"*, In: GECCO '21 Companion, Genetic and Evolutionary Computation Conference Companion, July 2021. URL: https://doi.org/10.1145/3449726.3459583
