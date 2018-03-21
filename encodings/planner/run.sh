#!/bin/bash

INSTANCE1=gripper-round-1-strips/instances/instance-5.pddl

INSTANCE2=gripper-round-1-strips/instances/instance-11.pddl

INSTANCE3=blocks-strips-typed/instances/instance-31.pddl

INSTANCE4=ins3.lp

INSTANCE5=/home/klaus/Desktop/Work/Efficient-grounding/benchmarks/instances-easy/ipc-2000/domains/blocks-strips-typed/instances/instance-12.pddl

INSTANCE6=transport-sequential-satisficing/instances/instance-14.pddl

INSTANCE7=transport-sequential-satisficing/instances/instance-8.pddl

INSTANCE8=transport-sequential-satisficing/instances/instance-1.pddl

INSTANCE9=tidybot-sequential-satisficing/instances/instance-3.pddl

INSTANCE10=airport-nontemporal-strips/instances/instance-5.pddl


python -B runplanner.py $INSTANCE1 --translate --parallel=1 --shallow $@

