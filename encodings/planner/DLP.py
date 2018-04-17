from __future__ import print_function

import Generators
from time import time
import clingo

import sys
import os

#
# MEMORY USAGE (for Unix)
#
def memory_usage_t(key="VmSize"):

    # data
    proc_status = '/proc/%d/status' % os.getpid()
    scale = {'kB': 1024.0, 'mB': 1,
             'KB': 1024.0, 'MB': 1}

    # get pseudo file  /proc/<pid>/status
    try:
        t = open(proc_status)
        v = t.read()
        t.close()
    except:
        return -1  # non-Linux?

    # get key line e.g. 'VmSize:  9999  kB\n ...'
    i = v.index(key)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return -1  # invalid format?

    # return
    return int(float(v[1]) / scale[v[2]])


class DynamicLogicProgram(object):

    # base class

    def __init__(self, files, program="", options=[], clingo_options=[]):
        pass

    def start(self):
        pass

    # ground(n) grounds n steps
    # ground(i,j) grounds from i to j (both included)
    def ground(self, start, end=None):
        pass

    def assign_external(self, external, val):
        pass

    def release_external(self, external):
        pass

    def add(self, program, args, rules):
        pass

    def cleanup(self):
        pass

    def solve(self, on_model, assumptions=[]):
        pass

    def get_model_str(self, m, step):
        return " ".join([str(atom) for atom in m.symbols(shown=True)])


class DynamicLogicProgramBackend(DynamicLogicProgram):

    def __init__(self, files, program="", options=[], clingo_options=[]):

        t = time()

        # preprocessing

        self.mem = memory_usage_t()

        generator_class = self.get_generator_class()
        generator = generator_class(
            files = files,
            adds  = [("base", [], program)],
            parts = [("base", [])],
            options = clingo_options,
        )

        # start
        dlp_container = generator.run()

        print("mem used after generating: {}MB".format(memory_usage_t() - self.mem))

        # init
        self.offset = dlp_container.offset
        self.rules = dlp_container.rules
        self.weight_rules = dlp_container.weight_rules
        self.primed_externals = dlp_container.primed_externals
        self.normal_externals = dlp_container.normal_externals
        self.output = dlp_container.output
        self.output_facts = dlp_container.output_facts
        self.init = dlp_container.init
        # rest
        self.control = clingo.Control(clingo_options)
        self.steps = 0
        self.assigned_externals = {}

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = time() - t
        self._start_time = 0
        self._ground_time = 0
    
    def get_generator_class(self):
        return Generators.DLPGenerator

    def start(self):

        t = time()
        with self.control.backend() as backend:
            for atom in self.init:
                backend.add_rule([atom], [], False)

        self._start_time = time() - t

    # ground(n) grounds n steps
    # ground(i,j) grounds from i to j (both included)
    def ground(self, start, end=None):
        ### fuction without any memory adds
        t = time()

        # preprocess
        if end == None:
            end = self.steps + start
            start = self.steps + 1
        elif self.steps != start-1:
            raise Exception(GROUNDING_ERROR)
        self.steps = end
        # start
        print("DLP: rules grounded: {}".format(len(self.rules)))
        with self.control.backend() as backend:
            for step in range(start, end+1):

                offset = (step-1)*self.offset

                for rule in self.rules:

                    backend.add_rule(
                        [x+offset for x in rule[1]],
                        [x-offset if x <= 0 else x+offset for x in rule[2]],
                        rule[0]
                    )

                for rule in self.weight_rules:
                    backend.add_weight_rule(
                        [x+offset for x in rule[1]],
                        rule[2],
                        [(x+offset,y) if x  > 0 else (x-offset,y) for x, y in rule[3]],
                        rule[0]
                    )
                for symbol in self.normal_externals.keys():
                    self.assigned_externals[(step, symbol)] = -1

        self._ground_time += time() - t 

    def assign_external(self, clingo_symbol, value):
        if len(clingo_symbol.arguments) != 1:
            print("ERROR:Clingo symbol must have one int argument!")
            raise ValueError
        step = int(clingo_symbol.arguments[-1].number)
        symbol = clingo.Function(clingo_symbol.name, clingo_symbol.arguments[:-1])
        if value is None:
            self.assigned_externals.pop((step, symbol), None)
        else:
            self.assigned_externals[(step, symbol)] = 1 if value else -1

    def release_external(self, clingo_symbol):
        if len(clingo_symbol.arguments) != 1:
            print("ERROR:Clingo symbol must have one int argument!")
            raise ValueError
        step = int(clingo_symbol.arguments[-1].number)
        symbol = clingo.Function(clingo_symbol.name, clingo_symbol.arguments[:-1])
        self.assigned_externals.pop((step, symbol), None)
        with self.control.backend() as backend:
            backend.add_rule(
                [], [self.normal_externals[symbol]+(step*self.offset)], False
            )

    def get_answer(self, model, step):
        out = [("*",symbol) for symbol in self.output_facts]
        for i in range(step+1):
            for atom, symbol in self.output:
                if model.is_true(atom+(i*self.offset)):
                    out.append((i, symbol))
        return out

    def get_assumptions(self):
        
        return [(self.normal_externals[key[1]]+(self.offset*key[0]))*value
                for key, value in self.assigned_externals.items()]

    def cleanup(self):
        self.control.cleanup()

    def solve(self, on_model, assumptions=[]):
        return self.control.solve(on_model=on_model, assumptions=assumptions+self.get_assumptions())

    @property
    def statistics(self):
        return self.control.statistics

    @property
    def ground_time(self):
        return self._init_time, self._start_time, self._ground_time
    
    def print_model(self, m, step):
        print("Step: {}\n{}\nSATISFIABLE".format(step, " ".join(
                    ["{}:{}".format(x,y) for x,y in self.get_answer(m, step)]
                )))
    def get_model_str(self, m, step):

        return " ".join(["{}:{}".format(x,y) for x,y in self.get_answer(m, step)])

    def __str__(self):
        out = ""
        output_dict = {}
        for atom, symbol in self.output:
            output_dict[atom] = "#prev(" + symbol + ")"
            output_dict[atom + self.offset] = symbol
        for rule in self.rules:
            if rule[0]:
                out += "{"
            out += "; ".join(
                [output_dict.get(head, str(head)) for head in rule[1]]
            )
            if rule[0]:
                out += "}"
            if not len(rule[2]):
                out += ".\n"
                continue
            if rule[1]:
                out += " "
            out += ":- " 
            out += ", ".join(
                [output_dict.get(b, str(b)) for b in rule[2] if b  > 0] +
                ["not " + output_dict.get(
                    -b, str(-b)
                ) for b in rule[2] if b <= 0]
            )
            out += ".\n"
        for rule in self.weight_rules:
            if rule[0]:
                out += "{"
            out += "; ".join(
                [output_dict.get(head, str(head)) for head in rule[1]]
            )
            if rule[0]:
                out += "}"
            if not len(rule[3]):
                out += ".\n"
                continue
            if rule[1]:
                out += " "
            out += ":- {} #sum ".format(rule[2])
            out += "{"
            out += "; ".join(
                [str(w) + "," + output_dict.get(b, str(b)) + ": " +
                 output_dict.get(b,str(b)) for b,w in rule[3] if b  > 0] +
                [str(w) + "," + output_dict.get(-b, str(-b)) + ": not " +
                 output_dict.get(-b,str(-b)) for b,w in rule[3] if b  <= 0]
            )
            out += "}.\n"
        for symbol, _ in self.primed_externals.items():
            out += "#external {}.\n".format(str(symbol))
        for symbol, _ in self.normal_externals.items():
            out += "#external {}.\n".format(str(symbol))
        for symbol in self.output_facts:
            out += "{}.\n".format(symbol)
        return out


class DynamicLogicProgramBackendSimplified(DynamicLogicProgramBackend):

    def __init__(self, files, program="", options=[], clingo_options=[]):

        t = time()

        self.mem = memory_usage_t()

        # preprocessing
        generator_class = self.get_generator_class()
        generator = generator_class(
            files = files,
            adds  = [("base", [], program)],
            parts = [("base", [])],
            options = clingo_options,
            compute_cautious = True,
            compute_brave = True
        )

        # start
        dlp_container = generator.run()

        print("mem used after generating: {}MB".format(memory_usage_t() - self.mem))

        # init
        self.offset = dlp_container.offset
        self.rules = dlp_container.rules
        self.weight_rules = dlp_container.weight_rules
        self.primed_externals = dlp_container.primed_externals
        self.normal_externals = dlp_container.normal_externals
        self.output = dlp_container.output
        self.output_facts = dlp_container.output_facts
        self.init = dlp_container.init
        # rest
        self.control = clingo.Control(clingo_options)
        self.steps = 0
        self.assigned_externals = {}

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = time() - t
        self._start_time = 0
        self._ground_time = 0
    
    def get_generator_class(self):
        return Generators.DLPGeneratorSimplifier

class DynamicLogicProgramBackendSimplified_NCNB(DynamicLogicProgramBackend):

    def __init__(self, files, program="", options=[], clingo_options=[]):

        t = time()

        # preprocessing
        generator_class = self.get_generator_class()
        generator = generator_class(
            files = files,
            adds  = [("base", [], program)],
            parts = [("base", [])],
            options = clingo_options,
            compute_cautious = False,
            compute_brave = False
        )

        # start
        dlp_container = generator.run()


        # init
        self.offset = dlp_container.offset
        self.rules = dlp_container.rules
        self.weight_rules = dlp_container.weight_rules
        self.primed_externals = dlp_container.primed_externals
        self.normal_externals = dlp_container.normal_externals
        self.output = dlp_container.output
        self.output_facts = dlp_container.output_facts
        self.init = dlp_container.init
        # rest
        self.control = clingo.Control(clingo_options)
        self.steps = 0
        self.assigned_externals = {}

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = time() - t
        self._start_time = 0
        self._ground_time = 0
    
    def get_generator_class(self):
        return Generators.DLPGeneratorSimplifier

class DynamicLogicProgramBackendClingoPre(DynamicLogicProgramBackend):

    def __init__(self, files, program="", options=[], clingo_options=[]):

        t = time()

        # preprocessing
        generator_class = self.get_generator_class()
        generator = generator_class(
            files = files,
            adds  = program,
            options = clingo_options
        )

        # start
        dlp_container = generator.run()


        # init
        self.offset = dlp_container.offset
        self.rules = dlp_container.rules
        self.weight_rules = dlp_container.weight_rules
        self.primed_externals = dlp_container.primed_externals
        self.normal_externals = dlp_container.normal_externals
        self.output = dlp_container.output
        self.output_facts = dlp_container.output_facts
        self.init = dlp_container.init
        # rest
        self.control = clingo.Control(clingo_options)
        self.steps = 0
        self.assigned_externals = {}

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = time() - t
        self._start_time = 0
        self._ground_time = 0
    
    def get_generator_class(self):
        return Generators.DLPGeneratorClingoPre


class DynamicLogicProgramText(DynamicLogicProgram):

    def __init__(self, files, program="", options={}, clingo_options=[]):

        t = time()

        self.grounder = Generators.Grounder(files=files, program=program, options=clingo_options)

        self.control = clingo.Control(clingo_options)

        self.step = 0

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = time() - t

        self._start_time = 0
        self._ground_time = 0

    def start(self):

        t = time()

        self.grounder.parse_prog()

        # add instance #show statements
        self.control.add("base", [], "\n".join(self.grounder.other_rules))

        # add initial state
        for atom in self.grounder.init:
            self.control.add("base", [], atom + ".")

        # add temporary constraint for time 0
        self.add(BASE, [], "#external query(0).")
        self.add(BASE, [], ":- not goal(0), query(0).")

        self.control.ground([("base", [])])

        self._start_time += time() - t
    

    # ground(n) grounds n steps
    # ground(i,j) grounds from i to j (both included)
    def ground(self, start, end=None):

        t = time()

        # preprocess
        if end == None:
            end = self.step + start
            start = self.step + 1
        elif self.step != start - 1:
            raise Exception("Error: Invalid grounding steps")
        self.step = end

        print("DLP: grounding from {} to {}".format(start, end))

        prog_name = "step{}".format(end)

        for timeg, rules in self.grounder.ground_range(start, end).iteritems():
            self.control.add(prog_name, [], "\n".join(rules))

        self.control.ground([(prog_name, [])])

        self._ground_time += time() - t

    def assign_external(self, external, val):
        #print("assigning external: ", str(external), str(val))
        self.control.assign_external(external, val)

    def release_external(self, external):
        #print("releasing external: ", str(external))
        self.control.release_external(external)

    def add(self, program, args, rules):
        self.control.add(program, args, rules)

    def cleanup(self):
        self.control.cleanup()

    def solve(self, on_model, assumptions=[]):
        return self.control.solve(on_model=on_model, assumptions=assumptions)

    @property
    def statistics(self):
        return self.control.statistics

    @property
    def ground_time(self):
        return self._init_time, self._start_time, self._ground_time

    def print_model(self, m, step):

        for atom in m.symbols(shown=True):
            print(atom)

    def __str__(self):
        return self.grounder.__str__()

BASE  = "base"
STEP  = "step"
CHECK = "check"
QUERY = "query"
SKIP  = "skip"

class DynamicLogicProgramBasic(DynamicLogicProgram):

    def __init__(self, files, program="", options=[], clingo_options=[]):

        t = time()

        self.control = clingo.Control(clingo_options)

        for f in files:
            self.control.load(f)

        if not program == "":
            self.control.add("base", [], program)

        self.step = 0

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = time() -t
        self._start_time = 0
        self._ground_time = 0

    def start(self):
        t = time()

        self.control.ground([(BASE, []), (CHECK, [0])])

        self._start_time = time() - t

    # ground(n) grounds n steps
    # ground(i,j) grounds from i to j (both included)
    def ground(self, start, end=None):

        t1 = time()

        # preprocess
        if end == None:
            end = self.step + start
            start = self.step + 1
        elif self.step != start - 1:
            raise Exception("Error: Invalid grounding steps")
        self.step = end

        print("DLP: grounding from {} to {}".format(start, end))

        parts = [(STEP, [t]) for t in range(start, end+1)]
        # parts = parts + [(CHECK,[length])]
        parts += [(CHECK, [t]) for t in range(start, end+1)]

        self.control.ground(parts)

        self._ground_time += time() - t1 

    def assign_external(self, external, val):
        self.control.assign_external(external, val)

    def release_external(self, external):
        self.control.release_external(external)

    def add(self, program, args, rules):
        self.control.add(program, args, rules)

    def cleanup(self):
        self.control.cleanup()

    def solve(self, on_model, assumptions=[]):
        return self.control.solve(on_model=on_model, assumptions=assumptions)

    @property
    def ground_time(self):
        return self._init_time, self._start_time, self._ground_time

    @property
    def statistics(self):
        return self.control.statistics

    def print_model(self, m, step):

        for atom in m.symbols(shown=True):
            print(atom)

def incmode():
    

    #dlp = DynamicLogicProgramBackend(["example.lp"], clingo_options=sys.argv[1:])
    #dlp = DynamicLogicProgramBackend(["example.lp"], clingo_options=sys.argv[1:])

    dlp = DynamicLogicProgramBackendClingoPre(["example-clingo-pre.lp"], clingo_options=sys.argv[1:])
    dlp.start()
    print(dlp); return

    # loop
    step, ret = 1, None
    while True:
        if step > 2: return
        dlp.release_external(clingo.Function("query",[step-1]))
        dlp.ground(1)
        dlp.assign_external(clingo.Function("query",[step]), True)

        with dlp.control.solve(assumptions=dlp.get_assumptions(), yield_ = True) as handle:
            for m in handle:
                print("Step: {}\n{}\nSATISFIABLE".format(step, " ".join(
                    ["{}:{}".format(x,y) for x,y in dlp.get_answer(m, step)]
                )))
                return
            print("Step: {}\nUNSATISFIABLE".format(step))
        step += 1

if __name__ == "__main__":
    incmode()

