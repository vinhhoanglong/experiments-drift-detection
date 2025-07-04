import math 
import numpy as np
import scipy.stats as stats
from typing import List, Tuple 
from scipy.signal import find_peaks
from enum import Enum
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes 
from tqdm import tqdm

from src.utils.helpers import _getActivityNames, _getActivityNames_LogList, makeProgressBar

def _getCausualFootprint(log:EventLog, activities:List[str]=None, activityName_key:str=xes.DEFAULT_NAME_KEY)-> np.chararray:
    if activities is None:
        activities = _getActivityNames(log, activityName_key=activityName_key)
    d = {(act1, act2): '%' for act1 in activities for act2 in activities}
    
    for trace in log:
        seen = set()
        A_touched = set()
        for event in trace:
            name = event[activityName_key]
            for s in seen:
                valnow = d[(s, name)]
                if valnow in ['%', 'A']:
                    d[(s,name)] = 'A'
                    A_touched.add((s,name))
                elif valnow == 'N':
                    d[(s, name)] = 'S'
            seen.add(name)
        for act1 in seen: 
            for act2 in activities:
                if d[(act1, act2)] == 'A' and (act1, act2) not in A_touched:
                    d[(act1, act2)] = 'S'
                elif d[(act1, act2)] == '%':
                    d[(act1, act2)] = 'N' 
    output = np.chararray((len(activities), len(activities)), unicode = True)
    for act1 in activities:
        i1 = activities.index(act1)
        for act2 in activities:
            i2 = activities.index(act2)
            output[i1][i2] = d[(act1, act2)] if d[(act1, act2)] != '%' else 'N'
    return output

