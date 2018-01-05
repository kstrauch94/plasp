#!/bin/bash

INSTANCE1=gripper-round-1-strips/instances/instance-3.pddl

INSTANCE2=gripper-round-1-strips/instances/instance-4.pddl

INSTANCE3=blocks-strips-typed/instances/instance-12.pddl

INSTANCE4=ins3.lp

INSTANCE5=/home/klaus/Desktop/Work/Efficient-grounding/benchmarks/instances-easy/ipc-2000/domains/blocks-strips-typed/instances/instance-12.pddl

INSTANCE6=transport-sequential-satisficing/instances/instance-14.pddl

INSTANCE7=transport-sequential-satisficing/instances/instance-8.pddl

# basic
#python -B runplanner.py $INSTANCE6 --translate --parallel=1 --shallow $@

# text DLP
#python -B runplanner.py $INSTANCE6 --translate --dlp=text --parallel=1 --shallow $@
# backend DLP
python -B runplanner.py $INSTANCE7 --translate --dlp=backend --parallel=1 --shallow $@
# backend simplified DLP
#python -B runplanner.py $INSTANCE7 --translate --dlp=backend-simplified --parallel=1 --shallow $@
# backend simplified NCNB DLP
#python -B runplanner.py $INSTANCE7 --translate --dlp=backend-simplified-ncnb --parallel=1 --shallow $@

#python runplanner.py $INSTANCE1 --translate --parallel=0 --heuristic=Domain heuristic.lp --verbose
#python planner.py --forbid-actions $INSTANCE4 basic.lp --verbose #--output-debug=text

