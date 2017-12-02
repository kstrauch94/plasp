#!/bin/bash

INSTANCE1=gripper-round-1-strips/instances/instance-3.pddl

INSTANCE2=gripper-round-1-strips/instances/instance-4.pddl

INSTANCE3=blocks-strips-typed/instances/instance-12.pddl

INSTANCE4=ins3.lp

INSTANCE5=/home/klaus/Desktop/Work/Efficient-grounding/benchmarks/instances-easy/ipc-2000/domains/blocks-strips-typed/instances/instance-12.pddl

# basic
#python -B runplanner.py $INSTANCE2 --translate --parallel=5

# text DLP
python -B runplanner.py $INSTANCE1 --translate --dlp=text --parallel=5 --stats
# backend DLP
#python -B runplanner.py $INSTANCE1 --translate --dlp=backend --parallel=5
# backend simplified DLP
#python -B runplanner.py $INSTANCE1 --translate --dlp=backend-simplified --parallel=5
# backend simplified NCNB DLP
#python -B runplanner.py $INSTANCE1 --translate --dlp=backend-simplified-ncnb --parallel=5

#python runplanner.py $INSTANCE1 --translate --parallel=0 --heuristic=Domain heuristic.lp --verbose
#python planner.py --forbid-actions $INSTANCE4 basic.lp --verbose #--output-debug=text

