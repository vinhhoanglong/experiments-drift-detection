#TODO: define the metrics

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD



def getTP_FP(detected:List[int], known:List[int], lag:int, countDuplicates:bool=False) -> Tuple[int, int]:
    """
    """
    assigments = assign_changepoints(detected, known, lag)
    TP = len(assigments)
    if countDuplicates:
        FP = len(detected) - TP
    else:
        TPCandidates = [d for d in detected if any((k - lag <= d and d <= k + lag) for k in known)]
        FP = len(detected) - len(TPCandidates)
    return (TP, FP)

def assign_changepoints(detected_changepoints: List[int], actual_changepoints:List[int], lag_window:int=200) -> List[Tuple[int,int]]:
    """Assigns detected changepoints to actual changepoints using a LP.
    With restrictions: 

    - Detected point must be within `lag_window` of actual point. 
    - Detected point can only be assigned to one actual point.
    - Every actual point can have at most one detected point assigned. 

        This is done by first optimizing for the number of assignments, finding how many detected change points could be assigned, without minimizing the \
        total lag. Then, the LP is solved again, minimizing the sum of squared lags, while keeping the number of assignments as high as possible.

    Args:
        detected_changepoints (List[int]): List of locations of detected changepoints.
        actual_changepoints (List[int]): List of locations of actual changepoints.
        lag_window (int, optional): How close must a detected change point be to an actual changepoint to be a true positive. Defaults to 200.

    Examples:
    >>> detected_changepoints = [1050, 934, 2100]
    >>> actual_changepoints = [1000,1149,2000]
    >>> assign_changepoints(detected_changepoints, actual_changepoints, lag_window=200)
    >>> [(1050, 1149), (934, 1000), (2100, 2000)]
    >>> # Notice how the actual changepoint 1000 gets a further detected changepoint to allow 1149 to also get a changepoint assigned

    Returns:
        List[Tuple[int,int]]: List of tuples of (detected_changepoint, actual_changepoint) assignments
    """

    def buildProb_NoObjective(sense):
        """
            Builds the optimization problem, minus the Objective function. Makes multi-objective optimization simple
        """
        prob = LpProblem("Changepoint_Assignment", sense)

        # Create a variable for each pair of detected and actual changepoints
        vars = LpVariable.dicts("x", (detected_changepoints, actual_changepoints), 0, 1, LpBinary) # Assign detected changepoint dp to actual changepoint ap?
        
        # Flatten vars into dict of tuples of keys
        x = {
            (dc, ap): vars[dc][ap] for dc in detected_changepoints for ap in actual_changepoints
        }

        ####### Constraints #########

        # Only assign at most one changepoint to each actual changepoint
        for ap in actual_changepoints:
            prob += (
                lpSum(x[dp, ap] for dp in detected_changepoints) <= 1,
                f"Only_One_Changepoint_Per_Actual_Changepoint : {ap}"
            )
        # Each detected changepoint is assigned to at most one actual changepoint
        for dp in detected_changepoints:
            prob += (
                lpSum(x[dp, ap] for ap in actual_changepoints) <= 1,
                f"Only_One_Actual_Changepoint_Per_Detected_Changepoint : {dp}"
            )
        # Distance between chosen changepoints must be within lag window
        for dp in detected_changepoints:
            for ap in actual_changepoints:
                prob += (
                    x[dp, ap] * abs(dp - ap) <= lag_window,
                    f"Distance_Within_Lag_Window : {dp}_{ap}"
                )
        return prob, x

    solver = PULP_CBC_CMD(msg=0)

    ### Multi-Objective Optimization: First maximize number of assignments to find out the best True Positive number that can be achieved
    # Find the largest number of change points:
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob1_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Maximize number of assignments"
    )
    prob1.solve(solver)
    # Calculate number of TP
    num_tp = len([
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob1_vars[dp, ap].varValue == 1
    ])


    ### Multi-Objective Optimization: Now minimize the squared distance between assigned changepoints, using this maximal number of assignments
    # Use this number as the number of assignments for second optimization
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob2_vars[dp, ap] * pow(dp - ap,2)
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Squared_Distances"
    )

    # Number of assignments is the number of true positives we found in the first optimization
    prob2 += (
        lpSum(
            prob2_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ) == num_tp,
        "Maximize Number of Assignments"
    )
    prob2.solve(solver)
    return [
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob2_vars[dp, ap].varValue == 1
    ]
def calcPrecisionRecall(detected:List[int], known:List[int], lag:int, zeroDivision=np.nan, countDuplicates:bool=True) -> Tuple[float, float]:
    """
    """
    TP, FP = getTP_FP(detected, known, lag, countDuplicates)
    if(TP+FP > 0):
        precision = TP/(TP+FP)
    else:
        precision = zeroDivision
    if(len(known) > 0):
        recall = TP/len(known)
    else:
        recall = zero_division
    return (precision, recall)

def calF1Score(detected:List[int], known:List[int], lag:int, zeroDivision=np.nan, verbose: bool=False, countDuplicates:bool=True) -> float:
    """
    """
    TP, FP = getTP_FP(detected, known, lag, countDuplicates)
    try:
        precision = TP / (TP+FP)
        recall = TP / len(known)

        f1_score = (2*precision*recall)/(precision+recall)
        return f1_score
    except ZeroDivisionError:
        if verbose:
            print("Calculation of F1-Score resulted in division by 0.")
        return zeroDivision

    pass


def test():
    detected = [700, 1290, 2299, 3543, 4768]
    actual = [1199, 2399, 3599, 4799]
    lag = 200
    print(getTP_FP(detected, actual, lag))
    print(calcPrecisionRecall(detected, actual, lag))
    print(calF1Score(detected, actual, lag))

if __name__ == "__main__":
    test()

