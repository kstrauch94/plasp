import subprocess
import time as timemodule
import clingo

from tempfile import NamedTemporaryFile
import os


time_str = "%times for transition \n#program base. \n time(1000000001)."

time_0 = "1000000000"
time_1 = "1000000001"

verbose = True
class Grounder(object):

    def __init__(self, files, program="", debug=False):

        self.debug = debug
        self.debug_gf_name = "g.lp"
        # write program + time to tempfile
        self.prog_file = NamedTemporaryFile(delete=False)

        self.prog_file.write(program)
        self.prog_file.write(time_str)
        self.prog_file.flush()

        files += [self.prog_file.name]

        # ground file as temp file
        self.gf = NamedTemporaryFile(delete=False)

        self.ground_command = "clingo {} --text > {}".format(" ".join(files), self.gf.name)

        self.grounded_rules = []
        self.other_rules = []
        self.init = []

        self.timelimit = None

        self.has_grounded = False

        self.grounding_time = 0


    def parse_prog(self):

        if self.has_grounded:
            print("The grounding process has already been done, skipping...")
            return

        start_time = timemodule.time()

        subprocess.call(self.ground_command, shell=True)
        self.prog_file.close()
        os.unlink(self.prog_file.name)

        main_criteria = [time_0, time_1]
        other_rule_criteria = ["timelimit", "#show"]

        with open(self.gf.name, "r") as f:
            for rule in f.readlines():
                if rule.strip() == "time(" + time_0 + ")." or rule.strip() == "time(" + time_1 + ").":
                    continue
                elif any(check in rule for check in other_rule_criteria):
                    if "timelimit(" in rule:
                        self.timelimit = int(rule[rule.find("(") + 1:rule.rfind(")")])
                    else:
                        self.other_rules.append(rule)
                elif "#external" in rule and time_1 in rule:
                    self.grounded_rules.append(rule)
                elif "init(" in rule:
                    atom = rule[rule.find("(")+1:rule.rfind(")")].replace(time_0, "0")
                    self.init.append(atom)
                elif "#external" not in rule and any(check in rule for check in main_criteria):
                    self.grounded_rules.append(rule)

        self.has_grounded = True

        if self.debug:
            with open(self.gf.name, "r") as f:
                with open(self.debug_gf_name, "w") as df:
                    df.write(f.read())

        self.gf.close()
        os.unlink(self.gf.name)

        self.grounding_time += timemodule.time() - start_time

    def ground_range(self, start, end):

        start_time = timemodule.time()

        if not self.has_grounded:
            print "ERROR: grounding has not taken place yet"
            return

        rules = {}

        for t in range(start, end+1):
            rules[t] = [rule.replace(time_0, str(t - 1)).replace(time_1, str(t)) for rule in self.grounded_rules]

        self.grounding_time += timemodule.time() - start_time

        return rules

    def ground(self, time):

        start_time = timemodule.time()

        if not self.has_grounded:
            print "ERROR: grounding has not taken place yet"
            return

        rules = {}

        rules[time] = [rule.replace(time_0, str(time - 1)).replace(time_1, str(time)) for rule in self.grounded_rules]

        self.grounding_time += timemodule.time() - start_time

        return rules

    def ground_to_file(self, start, end, filename="grounded.lp"):

        if not self.has_grounded:
            self.parse_prog()

        with open(filename, "w") as f:
            f.writelines(self.other_rules)
            f.writelines([init + ".\n" for init in self.init])
            for t, r in self.ground_range(start, end).iteritems():
                f.writelines(r)

    def __str__(self):
        string = ""
        for rule in self.other_rules:
            string += rule
        for rule in self.grounded_rules:
            string += rule

        return string

class DynamicLogicProgram(object):

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

class DynamicLogicProgramText(DynamicLogicProgram):

    def __init__(self, files, program="", options={}, clingo_options=[]):

        t = timemodule.time()

        self.grounder = Grounder(files=files, program=program)

        self.control = clingo.Control(clingo_options)

        self.step = 0

        # set solving and restart policy
        if 'restarts_per_solve' in options:
            self.control.configuration.solve.solve_limit = "umax," + str(options['restarts_per_solve'])

        if 'conflicts_per_restart' in options:
            if int(options['conflicts_per_restart']) != 0:
                self.control.configuration.solver[0].restarts = "F," + str(options['conflicts_per_restart'])

        self._init_time = timemodule.time() - t

        self._start_time = 0
        self._ground_time = 0

    def start(self):

        t = timemodule.time()

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

        self._start_time += timemodule.time() - t
    

    # ground(n) grounds n steps
    # ground(i,j) grounds from i to j (both included)
    def ground(self, start, end=None):

        t = timemodule.time()

        # preprocess
        if end == None:
            end = self.step + start
            start = self.step + 1
        elif self.step != start - 1:
            raise Exception("Error: Invalid grounding steps")
        self.step = end

        print("DLP: grounding from {} to {}".format(start, end))

        prog_name = "step{}".format(end)

        for time, rules in self.grounder.ground_range(start, end).iteritems():
            self.control.add(prog_name, [], "\n".join(rules))

        self.control.ground([(prog_name, [])])

        self._ground_time += timemodule.time() - t

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
            print atom

    def __str__(self):
        return self.grounder.__str__()

BASE  = "base"
STEP  = "step"
CHECK = "check"
QUERY = "query"
SKIP  = "skip"

class DynamicLogicProgramBasic(DynamicLogicProgram):

    def __init__(self, files, program="", options=[], clingo_options=[]):

        t = timemodule.time()

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

        self._init_time = timemodule.time() -t
        self._start_time = 0
        self._ground_time = 0

    def start(self):
        t = timemodule.time()

        self.control.ground([(BASE, []), (CHECK, [0])])

        self._start_time = timemodule.time() - t

    # ground(n) grounds n steps
    # ground(i,j) grounds from i to j (both included)
    def ground(self, start, end=None):

        t = timemodule.time()

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

        self._ground_time += timemodule.time() - t

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
            print atom



if __name__ == "__main__":
    g = Grounder(["basic.lp", "ins2.lp"])
    g.parse_prog()

    g.ground_to_file(1, 3)
