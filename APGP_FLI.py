# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 00:24:30 2018
nls-
@author: ID915897
random seed to ensure
"""
# Initialise
# Set number of runs
runs = 50

replacement = ''
# replacement = 'worst'


useFLI = 'yes'
# useFLI = 'no'

import csv
import itertools
import operator
import math
import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import gp
import datetime
import time
from math import exp, cos, sin, log
import pandas as pd
import numpy as np
import os
from functools import reduce
from operator import add, itemgetter
from numpy import arcsinh
from multiprocessing.pool import ThreadPool, threading

run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M-%S") #--------------------------------------------------------------(( 1 ))

try:
    devicename = os.environ['COMPUTERNAME']
    if devicename == 'DESKTOP-MLNSBQ2':
        system = 'laptop'
    if devicename == 'DESKTOP-JAN9GCB':
        system = 'NBtop'
    else: system = 'desktop'
except KeyError:
    system = 'server'  



"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""
problm = 'Concrete'
if system == 'server':
    with open("/home/aliyu/Documents/CORR/concrete_strength/Concrete_Data.csv") as train:
        trainReader = csv.reader(train)
        Tpoints = list(list(float(item) for item in row) for row in trainReader)
elif system == 'desktop':
    with open("C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Phase3\\Concrete_Strength\\Concrete_Data.csv") as train:
        trainReader = csv.reader(train)
        Tpoints = list(list(float(item) for item in row) for row in trainReader)
elif system == 'laptop':
    with open("C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Phase3\\Concrete_Strength\\Concrete_Data.csv") as train:
        trainReader = csv.reader(train)
        Tpoints = list(list(float(item) for item in row) for row in trainReader)
elif system == 'NBtop':
    with open("C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Phase3\\Concrete_Strength\\Concrete_Data.csv") as train:
        trainReader = csv.reader(train)
        Tpoints = list(list(float(item) for item in row) for row in trainReader)		

#split data: random selection without replacement
#Tpoints = points.copy()
random.seed(2019)   
x1=random.shuffle(Tpoints)   
split = int(len(Tpoints)*0.2)
datatrain=Tpoints[split:len(Tpoints)]
datatest=Tpoints[0:split]





"""
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
# The Asynchronous Parallel Steady State GP
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
"""
def mgpSteadyState(population, toolbox, cxpb, mutpb, ngen, poolsize, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None ):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    
    # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW 
    # get probability of evaluations and factor it in the number of breeds initiated
    # This will allow race to continue without stopping to check.
    factor = 1/((cxpb + mutpb) - cxpb*mutpb)  #--------------------------------***
	
	
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    target = 0.005							#--------------------------------------------------------------(())
    mettarget = 0 # 0 = not set
    trtarget = 0.005
    trmettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])

	#++++++++++++++++++++++++++++++++++++++++++++++++++
	#Evaluation of Initial Population
	#++++++++++++++++++++++++++++++++++++++++++++++++++
    # Evaluate the individuals with an invalid fitness       
    for ind in population:
        if not ind.fitness.valid:
            xo, yo, zo = toolbox.evaluate(ind, datatrain, datatest)
            ind.evlntime = yo,
            ind.testfitness = zo,
            ind.fitness.values = xo,
            if ind.fitness.values == (0.0101010101010101,) :
                ind.fitness.values = 0.0, #for maximising
            if ind.testfitness == (0.0101010101010101,) :
                ind.testfitness = 0.0, #for maximising                
#                print('check this out')
#                print(str(ind))
#                print(str(ind.fitness.values))
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
#    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0])])
    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                   halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])

    # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW
    gen=70
    #``````````````````````````````````````````````````````````````````````````````
    
	#+++++++++++++++++++++++++++++++++++++++++++++
	#+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
        return chosen
        
	#+++++++++++++++++++++++++++++++++++++++++++++
	#Breeding Function
	#+++++++++++++++++++++++++++++++++++++++++++++
	# define a breed function as nested.
    def breed():                                                                                    #, target, mettarget -----------???????
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, poolsize, target, mettarget, trtarget, trmettarget

        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values

        #++++++++ mutation on the offspring ++++++++++++++++               
        if random.random() < mutpb:
            p1, = toolbox.mutate(p1)
            del p1.fitness.values

        # Evaluate the offspring if it has changed
        if not p1.fitness.valid:
            #++++++++ Counting evaluations +++++++++++++++++
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations

            # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW            # NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW
            if counteval % poplnsize == 0:
#                try:
#                    halloffame.update(population)
#                except AttributeError:
#                    pass  
#                genStatsAPGP()
                genHof()

#                print(f'{counteval} evaluations initiated -- 	{round((100*counteval)/(ngen*poplnsize),2)}% of run {run}')
            # ````````````````````````````````````````````````````````````````````````````````````````````````````````

            counteval_lock.release()
            xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
            p1.evlntime = yo,
            p1.testfitness = zo,
            p1.fitness.values = xo, 
            #Check if ZeroDivisionError, ValueError 
            if p1.fitness.values == (0.0101010101010101,) :
                p1.fitness.values = 0.0, #for maximising
            if p1.testfitness == (0.0101010101010101,) :
                p1.testfitness = 0.0, #for maximising  

			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
			#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
            if float(p1.testfitness[0]) >= target:
#                print('Hi')
                if mettarget == 0:
                    mettarget = counteval
                    print(f'Target met: {counteval}')
                    print(f'Test Fitness: {float(p1.testfitness[0])}')
                    targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.testfitness[0]), 'Met_at': mettarget}, index = {run})
                
                    target_csv = f'{report_csv[:-4]}_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(target_csv):
                        targetmet_df.to_csv(target_csv, mode='a', header=False)
                    else:
                        targetmet_df.to_csv(target_csv)                    

            if float(p1.fitness.values[0]) >= trtarget:
#                print('Hi')
                if trmettarget == 0:
                    trmettarget = counteval
                    print(f'Training Target met: {trmettarget}')
                    print(f'Training Fitness: {float(p1.fitness.values[0])}')
                    # targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    trtargetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': trmettarget}, index = {run})
                
                    trtarget_csv = f'{report_csv[:-4]}_TRN_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(trtarget_csv):
                        trtargetmet_df.to_csv(trtarget_csv, mode='a', header=False)
                    else:
                        trtargetmet_df.to_csv(trtarget_csv) 
			#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]   
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END

        #+++++++++++++++++++++++++++++++++++++++++++++
        #Identify an individual to be replaced
        #+++++++++++++++++++++++++++++++++++++++++++++
        if replacement == 'worst': # Worst from fitness from population
            update_lock.acquire()            # LOCK !!!
            candidate = toolbox.worstfitness(population,1)[0]
            population.append(p1) 
            population.remove(candidate)
            update_lock.release()            # RELEASE !!!
	
        else: # Worst from random selection
            update_lock.acquire()            # LOCK !!!  
            # Identify a individual to replace - Worst from tournament(i.e. Inverse Tournament)
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
            update_lock.release()            # RELEASE !!!
			#+++++++++++++++++++++++++++++++++++++++++++++

		#Update hall of fame   ???? ==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass 
 
	##############################################################################   
	##############################################################################   
	##############################################################################   
    
    def genStatsAPGP():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv, update_lock
        #++++++++++ Collect Stats ++++++++++++++++++++
        update_lock.acquire() 
        record = stats.compile(population) if stats else {}
        update_lock.release() 
        
        logbook.record(run= run, gen=round(counteval/500), nevals=counteval, **record)
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, round(counteval/500), str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                       halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
      
#        if verbose:
#            print(logbook.stream) 

    def genHof():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv, update_lock
        #++++++++++ Collect Stats ++++++++++++++++++++
#        update_lock.acquire() 
#        record = stats.compile(population) if stats else {}
#        update_lock.release() 
        
#        logbook.record(run= run, gen=round(counteval/500), nevals=counteval, **record)
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, round(counteval/500), str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                       halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
      

	##############################################################################   
	##############################################################################   
	##############################################################################        
	##############################################################################      
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        #=============Update HOF Evaluation Time - Outside Thread =============
        #Capture Evaluation Time of HOF outside the threading
        xo, yo, zo = toolbox.evaluate(halloffame[0], datatrain, datatest)
        halloffame[0].evlntime = yo,
        #===========================================================
                
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                       halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
        
        #+++++++ END OF GENERATION +++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++

    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 
		#+++++++++++++++++++++++++++++++++++++++++++++
		#Create Report for the Run 
		#+++++++++++++++++++++++++++++++++++++++++++++
        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        
		#=============Update HOF Evaluation Time  ====================????????????
		#HOF Evaluation Time update - outside the threading
        for j in range(len(halloffame)):
            xo, yo, zo = toolbox.evaluate(halloffame[j], datatrain, datatest)
            halloffame[j].evlntime = yo,
		#===========================================================
    
		#+++++++++++++++++++++++++++++++++++++++++++++
		## Save 'Hall Of Fame' database
		#++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)


	# NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW NW
	##+++++++++++++++++++++++++++++++++++++++++++++
	## BEGIN RACE 
	##+++++++++++++++++++++++++++++++++++++++++++++
    tp = ThreadPool(poolsize)  # Set Degree of concurrency
    # Determine number of evaluations based on population size and number of generations:
    poplnsize =  len(population)
    targetevalns = poplnsize*ngen
    counteval = 0 # initialise for monitoring evaluation count
    print(f'Thread count: {threading.active_count()}')

	# The probabilities of mutation and crossover are factored to determine how many evaluations to do.  
    for h in range(int(poplnsize*ngen*factor)+100):
        tp.apply_async(breed)

	#   Wait for all jobs to complete and close threads.
    tp.close() # <-----------------------------------------------??????????????
    tp.join() #  <-----------------------------------------------??????????????

	# Feedback on screen
    print(f'done  : {counteval}')
    print(f'Target: {targetevalns}')
    print(threading.active_count())

	#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	# Collect Stats
#    collectStatsGen()
    collectStatsRun()
	# `````````````````````````````````````````````````````````````````````````````   
    
###############################################################################
###############################################################################       
    return population, logbook    
###############################################################################
"""    
#==============================================================================
#==============================================================================
"""




"""
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
# The Steady State GP
#+++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++
"""
def gpSteadyState(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    target = 0.005
    mettarget = 0 # 0 = not set
    trtarget = 0.005
    trmettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

	#+++++++++++++++++++++++++++++++++++++++++++++
	#Evaluation of Initial Population
	#+++++++++++++++++++++++++++++++++++++++++++++
    # Evaluate the individuals with an invalid fitness
	# ZeroDivisionError = 0.0101010101010101
    for ind in population:
        if not ind.fitness.valid:
            xo, yo, zo = toolbox.evaluate(ind, datatrain, datatest)
            ind.evlntime = yo,
            ind.testfitness = zo,
            ind.fitness.values = xo,
            if ind.fitness.values == (0.0101010101010101,) :
                ind.fitness.values = 0.0, #for maximising
            if ind.testfitness == (0.0101010101010101,) :
                ind.testfitness = 0.0, #for maximising                
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
    hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                   halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
	#+++++++++++++++++++++++++++++++++++++++++++++
	#+++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter

    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
        return chosen

	#+++++++++++++++++++++++++++++++++++++++++++++
	#Breeding Function
	#+++++++++++++++++++++++++++++++++++++++++++++
	# define a breed function as nested.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, target, mettarget, trtarget, trmettarget

        #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))

        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values

        #++++++++ mutation on the offspring ++++++++++++++++               
        if random.random() < mutpb:
            p1, = toolbox.mutate(p1)
            del p1.fitness.values

        # Evaluate the offspring if it has changed
        if not p1.fitness.valid:
            #++++++++ Counting evaluations +++++++++++++++++
            counteval_lock.acquire()
            counteval += 1 #Count the actual evaluations
            counteval_lock.release()
            xo, yo, zo = toolbox.evaluate(p1, datatrain, datatest)
            p1.evlntime = yo,
            p1.testfitness = zo,
            p1.fitness.values = xo, 
            #Check if ZeroDivisionError, ValueError 
            if p1.fitness.values == (0.0101010101010101,) :
                p1.fitness.values = 0.0, #for maximising
            if p1.testfitness == (0.0101010101010101,) :
                p1.testfitness = 0.0, #for maximising  

			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
			#[[[[[[[[[[[[[[[[[[[[ CHECK WHEN TARGET IS MET [[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]
			#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
            # if float(p1.fitness.values[0]) >= target:
            if float(p1.testfitness[0]) >= target:
#                print('Hi')
                if mettarget == 0:
                    mettarget = 500*(gen -1) + counteval
                    print(f'Target met: {mettarget}')
                    print(f'Test Fitness: {float(p1.testfitness[0])}')
                    # targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.testfitness[0]), 'Met_at': mettarget}, index = {run})
                
                    target_csv = f'{report_csv[:-4]}_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(target_csv):
                        targetmet_df.to_csv(target_csv, mode='a', header=False)
                    else:
                        targetmet_df.to_csv(target_csv) 
                        
            if float(p1.fitness.values[0]) >= trtarget:
#                print('Hi')
                if trmettarget == 0:
                    trmettarget = 500*(gen -1) + counteval
                    print(f'Training Target met: {trmettarget}')
                    print(f'Training Fitness: {float(p1.fitness.values[0])}')
                    # targetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': mettarget}, index = {run})
                    trtargetmet_df = pd.DataFrame({'Run' : run, 'Target': target, 'Fitness': float(p1.fitness.values[0]), 'Met_at': trmettarget}, index = {run})
                
                    trtarget_csv = f'{report_csv[:-4]}_TRN_Target.csv'
                    #Export from dataframe to CSV file. Update if exists
                    if os.path.isfile(trtarget_csv):
                        trtargetmet_df.to_csv(trtarget_csv, mode='a', header=False)
                    else:
                        trtargetmet_df.to_csv(trtarget_csv)   
			#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]   
			#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END

		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		# REPLACEMENT -  ++++++++++++++++++++++++++++++++++++++++++++
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Identify an individual to be replaced
        #+++++++++++++++++++++++++++++++++++++++++++++
        if replacement == 'worst': # Worst from fitness from population
#            update_lock.acquire()            # LOCK !!!
            candidate = toolbox.worstfitness(population,1)[0]
            population.append(p1) 
            population.remove(candidate)
#            update_lock.release()            # RELEASE !!!
	
        else: # INVERSE TOURNAMENT (Worst from random selection)       
#            update_lock.acquire()            # LOCK !!!  
            # Identify a individual to replace - Worst from tournament(i.e. Inverse Tournament)
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            if p1.fitness.values[0] > candidate.fitness.values[0]: # Max
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                population.append(p1) 
                population.remove(candidate)
#            update_lock.release()            # RELEASE !!!
			#+++++++++++++++++++++++++++++++++++++++++++++
        
#
		#    Update hall of fame   ????==== INDENT TO HAPPEN ONLY IF A RELPLACEMENT IS DONE ====?????                                                                     
        try:
            halloffame.update(population)
        except AttributeError:
            pass 

	################################################################################        
    def collectStatsGen():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
        #++++++++++ Collect Stats ++++++++++++++++++++
        record = stats.compile(population) if stats else {}
        logbook.record(run= run, gen=gen, nevals=counteval, **record)
        
        if verbose:
            print(logbook.stream) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Update hall of fame database for each generation
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                       halloffame[0].evlntime, len(halloffame[0]), str(halloffame[0])])
        
        #+++++++ END OF GENERATION +++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++++++++++++
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

		#+++++++++++++++++++++++++++++++++++++++++++++
			#Create Report for the Run 
		#+++++++++++++++++++++++++++++++++++++++++++++
        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        data = np.array([[*a, *b, *c, *d] for a, b, c, d in zip(*data)])
        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        
		#++++++++++++++++++++++++++++++++++++++++++++++
		## Save 'Hall Of Fame' database
		#++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
        #Destination file (local)
        hof_csv = f'{report_csv[:-4]}_hof.csv'
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)

	#+++++++++++++++++++++++++++++++++++++++++++++
	#Create a Generation
	#+++++++++++++++++++++++++++++++++++++++++++++
	# Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
        counteval = 0 
        for h in range(poplnsize):
            breed()
        # Work with number of actual evaluations done.
		# If expected number of evaluations not met continue
        while counteval < poplnsize:
            for j in range(poplnsize - counteval):
                breed()

        collectStatsGen()
    collectStatsRun()

	############################################################################       
    return population, logbook    
	############################################################################

"""    
#==============================================================================
#==============================================================================
"""

if system == 'server':
    result_dir = f'/home/aliyu/Documents/FL_APGP/Concrete/'
elif system == 'desktop':	
    result_dir = f'C:\\Users\\ID915897\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_APGP\\Concrete\\fli_set4_nolock\\'
elif system == 'laptop':
    result_dir = f'C:\\Users\\aliyus\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_APGP\\Concrete\\fli_set4_nolock\\'
elif system == 'NBtop':
    result_dir = f'C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Ph4_Evln_time\\FL_APGP\\Concrete\\fli_set4_nolock\\'

#--------------------------------------------------

"""    
#==============================================================================
#==============================================================================
"""
def div(left, right):
    return left / right

#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 13), float, "x")
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 5), float, "x") #Airfoil
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #Wine quality
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #Concrete

pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(div, [float,float], float) # ???????????????????
#pset.addPrimitive(math.log, [float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin,[float], float)

pset.addEphemeralConstant("nrand101c", lambda: random.randint(1,100)/20, float)  #(3a)
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Weight is positive (i.e. maximising problem)  for normalised.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)#-----------0000000000000000000000000000--------------if length greater than 30-------------------?
#toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)#
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)




#    Evaluate the mean squared error between the expression
#def evalSymbReg(individual, points):
def evalSymbReg(individual, datatrain, datatest):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    time_st = time.perf_counter() # data only
    # Evaluate the mean squared error between the expression and the real function
    #Training Error - Fitness ===============================
    for z in range(2): #                                                       (5)
        error=0.
        total=0.
        try:
            for item in datatrain:
                total = total + ((func(*item[:8])) - (item[8]))**2    #-------------Concrete_Strength
#                sqerrors = sum(((func(*item[:57])) - (item[57]))**2 for item in datatrain)
                MSE = total/len(datatrain)
                error = 1/(1+ MSE)               
#                ---
#                MSE = total/len(datatrain)
#                error = 1/(1+ MSE)
        except (ZeroDivisionError, ValueError):#    except ZeroDivisionError:
                error = 0.010101010101010101

    #Test Data =============================================
    error_test=0.
    total_t=0.
    try:
        for item in datatest:
            total_t = total_t + ((func(*item[:8])) - (item[8]))**2                                      #Dow_Chemicals
#            sqerrors_t = sum(((func(*item[:57])) - (item[57]))**2 for item in datatest)
            MSE_t = total_t/len(datatest)
            error_test = 1/(1+ MSE_t)               
#            ---
#            MSE_t = total_t/len(datatest) # mean squared error
#            error_test =1/(1+ MSE_t) # Normalise
    except (ZeroDivisionError, ValueError):#    except ZeroDivisionError:
            error_test = 0.010101010101010101
            
    evln_sec=float((time.perf_counter() - time_st))
    return error, evln_sec, error_test


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""

#============================================================
#============================================================
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3) # --------------------------?? breeding
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #Limit the size of a child
toolbox.register("worstfitness", tools.selWorst)
#===========================================================
#Function to collect stats for the last generation
def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None):
    lastgen_db=[]    
    for j in range(len(population)):
        xo, yo, zo = toolbox.evaluate(population[j], datatrain, datatest)
        population[j].fitness.values = xo,
        population[j].evlntime = yo,
        population[j].testfitness = zo,
        lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]), len(population[j]), str(population[j])])
    lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'Best'])
    #Destination file
    lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
    #Export from dataframe to CSV file. Update if exists
    if os.path.isfile(lastgen_csv):
        lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
    else:
        lastgen_dframe.to_csv(lastgen_csv)
#===========================================================
#===========================================================    



"""
============================================================================
Function to create initial population: (1) FIXED SIZE AND (2)  UNIQUE INDIVIDUALS
(Constants are treated as same).
============================================================================
"""
def inipopln(indlen=10):    
    ini_len = indlen # Initial lengths
    popsize = 500
    print(f'Creating a population of {popsize} individuals - each of size: {ini_len}')
    # Function to extract the node types   ----------------------------------------
    def graph(expr):
        str(expr)
        nodes = range(len(expr))
        edges = list()
        labels = dict()
        stack = []
        for i, node in enumerate(expr):
            if stack:
                edges.append((stack[-1][0], i))
                stack[-1][1] -= 1
            labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
            stack.append([i, node.arity])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return nodes, edges, labels
#    ------------------------------- create 1st individual
    current=[]
    newind=[]
    newind= toolbox.population(n=1)
    while len(newind[0]) != ini_len:
        newind = toolbox.population(n=1)
    current.append(newind[0])
#    ------------------------------- Create others; 
#    For each new one check to see a similar individual exists in the population.
    while len(current) < popsize:
        pop = toolbox.population(n=1)
        if len(pop[0]) == ini_len:
            # ----------------------------- Check for duplicate
            lnodes, ledges, llabels = graph(pop[0])
            similarity = 'same'
            for k in range(len(current)): # CHECK all INDs in CURRENT population
                nodes, edges, labels = graph(current[k])
                for j in range(len(labels)): # Check NEW against IND from CURRENT
                    constants = 'no' # will use to flag constants
                    if labels[j] != llabels[j]: 
                        similarity = 'different' 
                        # no need to check other nodes as soon as difference is detected 
                    if '.' in str(labels[j]) and '.' in str(llabels[j]): constants = 'yes'
                    if labels[j] != llabels[j] or constants != 'yes': # They are different and not constants
                        continue # no need to check other nodes as soon as difference is detected 
                if similarity =='same': # skips other checks as soon as it finds a match
                    continue
            if similarity == 'different': # add only if different from all existing
                current.append(pop[0])     
    print('population created')
    return current
"""
============================================================================
============================================================================
"""
    
def main():
    random.seed(2019)

    tag = f'_{problm}_stdGP_SteadyState' #Output file name prefix
    report_csv = f"{result_dir}{tag}_{run_time}.csv"

    if replacement == 'worst': report_csv = f"{result_dir}Rworst_{tag}_{run_time}.csv"

    for i in range(1, runs+1):
        run = i
        if useFLI == "yes":
            pop = inipopln()
        else:
            pop = toolbox.population(n=500)

        hof = tools.HallOfFame(1) 
        #-----------------------------------------       
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
        stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        pop, log = gpSteadyState(pop, toolbox, 0.9, 0.1, 70, 		#              (9)
                                  stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv)
        print('Taking stats for the last generation....')
        #Collect stats for the last generation of each run.
        lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv)	#GEN....??  (9b)


##============== Run APGP ===========================================
#    poolsizelist =[5, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
#    FLIlist = [5,10,15,20,25,30,35,40,45,50]
    FLIlist = [10]
#    run_time = '20200627_1354-43'
    for indlen in FLIlist:

        poolsizelist =[50]#]
#        poolsizelist =[5, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]#]
        for poolsize in poolsizelist:
            tag = f'FLIanaly{indlen}_{problm}_APGP_{poolsize}p' #Output file name prefix
            report_csv = f"{result_dir}{tag}_{run_time}.csv"
    
            if replacement == 'worst': report_csv = f"{result_dir}{tag}_Rworst_{run_time}.csv"
    
            for i in range(1, runs+1):        #=================???????????????????????
                run = i
                if useFLI == "yes":
                    pop = inipopln(indlen)
                else:
                    pop = toolbox.population(n=500)

                hof = tools.HallOfFame(1) 
                #-----------------------------------------       
                stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
                stats_size = tools.Statistics(len)
                stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
                stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
                mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness)
                mstats.register("avg", numpy.mean)
                mstats.register("std", numpy.std)
                mstats.register("min", numpy.min)
                mstats.register("max", numpy.max)
                pop, log = mgpSteadyState(pop, toolbox, 0.9, 0.1, 70, #              (9)       
                                          stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, poolsize=poolsize)
                print('Taking stats for the last generation....')
                #Collect stats for the last generation of each run.
                lastgenstats(pop, toolbox, gen=70, run=run, report_csv=report_csv)#GEN.......??  (9b)
    		

#==============================================================================
if __name__ == "__main__":
    main()    
#==============================================================================

