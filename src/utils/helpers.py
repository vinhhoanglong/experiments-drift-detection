#TODO: write helper functions 

import ast 
import datetime 
from pathlib import Path 
from typing import Any, List, Set, Tuple
import numpy 

import networkx as nx

from pm4py.util import xes_constants as xes
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from tqdm.auto import tqdm 

import pandas as pd

import datetime as dt
from datetime import timedelta 


def _dateToDatetime(date:dt.date) -> dt.datetime:
    # convert dt.date to dt.datetime 
    return dt.datetime(date.year, date.month, date.day)

def _getTimeDifference(time1:dt.datetime, time2: dt.datetime, scale:str) -> float:
    # return time gap between 2 dt object with scale: 'minutes', 'hours', 'days'
    
    duration_sec = (time2-time1).total_seconds()
    if scale == "minutes":
        return duration_sec / 60
    if scale == "hours":
        return duration_sec / 3600
    if scale == "days" :
        return (duration_sec / 3600) / 24
    return duration_sec

def _getNumActivities(log: EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->int:
    # calculate the number of activities in an event log
    return len(_getActivityNames(log, activityName_key))

def _getActivityNames(log: EventLog, activityName_key:str=xes.DEFAULT_NAME_KEY)->List[str]:
    #find distinct activities occuring in the event log
    
    return sorted(list({event[activityName_key] for case in log for event in case }))

def _getActivityNames_LogList(logs:List[EventLog], activityName_key:str=xes.DEFAULT_NAME_KEY)->List[str]:
    #find the distinct activities occuring in a list of log
    return sorted(list({ event[activityName_key] for log in logs for case in log for event in case}))

def makeProgressBar(num_iters:int=None, message:str="", position:int=None):
    return tqdm(total=num_iters, desc=f"{message} :: ", position=position, leave=True)

def safe_update_bar(progress_bar, amount:int=1)->None:
    if progress_bar is not None:
        progress_bar.update(amount)

def transitiveRedution(relation:Set[Tuple[Any,Any]]) -> Set[Tuple[Any,Any]]:
    digraph = nx.DiGraph(list(relation))
    reduction = nx.transitive_reduction(digraph)
    return set(reduction.edges)
