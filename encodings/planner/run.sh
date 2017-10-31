#!/bin/bash

INSTANCE1=gripper-round-1-strips/instances/instance-3.pddl

INSTANCE2=gripper-round-1-strips/instances/instance-4.pddl

INSTANCE3=blocks-strips-typed/instances/instance-12.pddl

INSTANCE4=ins3.lp


# text DLP
#python -B runplanner.py $INSTANCE1 --translate --parallel=0 --basic

# basic DLP
python -B runplanner.py $INSTANCE1 --translate --parallel=0 


#python runplanner.py $INSTANCE1 --translate --parallel=0 --heuristic=Domain heuristic.lp --verbose
#python planner.py --forbid-actions $INSTANCE4 basic.lp --verbose #--output-debug=text
