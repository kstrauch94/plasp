#!/usr/bin/python
from __future__ import print_function
import sys
import os
import argparse

from config import *

PLASP         = configuration["executables"]["plaspTranslate"]["command"]
PLANNER       = os.path.join(configuration["executables"]["planner"]["directory"], configuration["executables"]["planner"]["command"])
BASIC         = os.path.join(configuration["encodings"]["planner"]["directory"], configuration["encodings"]["planner"]["basic"])
PRE_SIMPLE    = os.path.join(configuration["encodings"]["planner"]["directory"], configuration["encodings"]["planner"]["preprocessSimple"])
HEURISTIC     = os.path.join(configuration["encodings"]["planner"]["directory"], configuration["encodings"]["planner"]["heuristic"])
PREPROCESS    = os.path.join(configuration["encodings"]["strips"]["directory"], configuration["encodings"]["strips"]["preprocess"])
STRIPS        = os.path.join(configuration["encodings"]["strips"]["directory"], configuration["encodings"]["strips"]["stripsIncremental"])
REDUNDANCY    = os.path.join(configuration["encodings"]["strips"]["directory"], configuration["encodings"]["strips"]["redundancy"])
POSTPROCESS   = os.path.join(configuration["encodings"]["strips"]["directory"], configuration["encodings"]["strips"]["postprocess"])
INCMODE       = os.path.join(configuration["encodings"]["strips"]["directory"], configuration["encodings"]["strips"]["incmode"])
TMP           = configuration["tmpFile"]
BASIC_OPTIONS = " " + configuration["basicOptions"] + " "

TEST_FILE     = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["test"])
TEST_FILE2    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["test_const"])
TEST_FILEM    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["test_model"])
TEST_MODEL    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_model"])
TEST_FORALL_1 = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_forall_1"])
TEST_FORALL_T = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_forall_t"])
TEST_B_FALL_1 = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_forall_1_basic"])
TEST_B_FALL_T = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_forall_t_basic"])
TEST_EXISTS_1 = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_1"])
TEST_EXISTS_T = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_t"])
TEST_B_EX_1   = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_1_basic"])
TEST_B_EX_T   = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_t_basic"])
TEST_EDGE_1   = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_edge_1"])
TEST_EDGE_T   = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_edge_t"])
TEST_B_EDGE_1 = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_edge_1_basic"])
TEST_B_EDGE_T = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_exists_edge_t_basic"])
TEST_SEQ_1    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_sequential_1"])
TEST_SEQ_T    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_sequential_t"])
TEST_DSEQ_1   = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_dyn_sequential_1"])
TEST_DSEQ_T   = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_dyn_sequential_t"])
TEST_ACT_1    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_actions_1"])
TEST_ACT_T    = os.path.join(configuration["testFiles"]["directory"], configuration["testFiles"]["block_actions_t"])

# Other systems
CLINGO      = "clingo"
#FAST_D      = os.path.join(configuration["executables"]["fastDownward"]["directory"], configuration["executables"]["fastDownward"]["command"])
#FAST_D_TR   = os.path.join(configuration["executables"]["fastDownwardTranslate"]["directory"], configuration["executables"]["fastDownwardTranslate"]["command"])

FAST_D      = "/home/klaus/bin/Fast-Downward/fast-downward.py --alias seq-sat-lama-2011"
FAST_D_TR   = "/home/klaus/bin/Fast-Downward/fast-downward.py --translate"

SAS_OUTPUT  = "output.sas"
M           = configuration["executables"]["madagascarM"]["command"]
MP          = configuration["executables"]["madagascarMp"]["command"]
MPC         = configuration["executables"]["madagascarMpC"]["command"]

#DLPs
DLP_BASIC   = "basic"
DLP_TEXT    = "text"
DLP_BACKEND = "backend"
DLP_BACKEND_SIMPLIFIED = "backend-simplified"
DLP_BACKEND_SIMPLIFIED_NCNB = "backend-simplified-ncnb"

#DLP encodings

DLP_BASIC_ENCODING = "basic.lp" # or strips inc?
DLP_TEXT_ENCODING  = "basic-text.lp"
DLP_BACKEND_ENCODING = "basic-backend.lp"


class MyArgumentParser:

    help = """
Planner and Clingo Options:
  --<option>[=<value>]\tSet planner or clingo <option> [to <value>]

    """
    usage = "runplanner.py instance [options]"
    epilog = " "
    epilog = """
Default command-line:
runplanner.py instance --closure=3 --parallel=3

runplanner.py is part of Potassco: https://potassco.org/labs
Get help/report bugs via : https://potassco.org/support
    """

    def run(self):

        # command parser
        _epilog = self.help + "\nusage: " + self.usage + self.epilog
        cmd_parser = argparse.ArgumentParser(description='Running an ASP-based PDDL Planner',
            usage=self.usage,epilog=_epilog,formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)

        # basic
        basic = cmd_parser.add_argument_group('Basic Options')
        basic.add_argument('-h','--help',action='help',help='Print help and exit')
        basic.add_argument('--dry-run',dest='dry_run',action='store_true',help='Print system call without running the planner')
        basic.add_argument('-p','--print-call',dest='print_call',action='store_true',help='Print system call before running the planner')
        #basic.add_argument('-',dest='read_stdin',action='store_true',help=argparse.SUPPRESS)
        #basic.add_argument('-c','--const',dest='constants',action="append",help=argparse.SUPPRESS,default=[])
        #basic.add_argument('-v','--verbose',dest='verbose',action="store_true",help="Be a bit more verbose")
        basic.add_argument('instance',help="PDDL instance, with corresponding domain file in the same directory (named either domain.pddl, or domain_instance), or defined via option --domain")
        basic.add_argument('--domain',dest='domain',help="PDDL domain",default=None)
        basic.add_argument('--hack',dest='hack',action='store_true',help=argparse.SUPPRESS)

        # specific
        normal = cmd_parser.add_argument_group('Solving Options')
        normal.add_argument('--translate','-tr',dest='translate',action='store_true',help='Run fast-downward translator to sas, then plasp translator, and solve')
        normal.add_argument('--closure',default=3,type=int,choices=[0,1,2,3],help='Static analysis of potentially relevant actions (default: 3)')
        normal.add_argument('--parallel',default=3,type=int,choices=[0,1,2,3,4,5],help='Sequential and parallel planning encoding variants (default: 3)')
        normal.add_argument('--shallow',action='store_true', help='Cheaply approximate mutually disabling parallel actions')
        normal.add_argument('--redundancy',action='store_true',help='Enforcement of redundant actions')
        normal.add_argument('--postprocess',action='store_true',help='Solve, serialize, and check if solution is correct (works also with --basic)')
        normal.add_argument('--preprocess-simple',action='store_true',help='Use simple preprocessing encoding')
        normal.add_argument('--use-heuristic', dest='use_heuristic', action='store_true',help='Run domain heuristic for planning')
        normal.add_argument('--test',default=None, type=int, choices=[0,1],
                            help="Test solution (0) using all non-serializable actions, or (1) using a minimal subset of them")
        normal.add_argument('--test-add', dest="test_add", default=5, type=int, choices=[0,1,2,3,4,5,6,7],
                            help="""Add constraints \
(0) deleting the model, or \ 
(1) inforcing forall plans, or \
(3) inforcing exists plans, or \
(4) inforcing exists plans with #edge directives, or \
(5, default) inforcing sequential plans, or \
(6) inforcing sequential plans (relaxed), or \
(7) deleting the non-serializable actions""")
        normal.add_argument('--test-times', dest = "test_times", default=1, type=int, choices=[0,1],
                            help="Add constraints (0) for action's times, or (1, default) for all times")

        extended = cmd_parser.add_argument_group('Other Solving Modes')
        extended.add_argument('--incmode',dest='incmode',action='store_true',help='Run clingo incmode')
        extended.add_argument('--basic',dest='basic',action='store_true',help='Run fast-downward translator to sas, then plasp translator, and solve with the basic encoding')
        extended.add_argument('--fast-downward','-fd',dest='fast-downward',action='store_true',help='Run fast-downward heuristic search planner with LAMA settings')
        extended.add_argument('--madagascar-M',  dest='M',  action='store_true',help='Run version   M of madagascar SAT planner')
        extended.add_argument('--madagascar-Mp', dest='Mp', action='store_true',help='Run version  Mp of madagascar SAT planner')
        extended.add_argument('--madagascar-MpC',dest='MpC',action='store_true',help='Run version MpC of madagascar SAT planner')

        extended.add_argument('--dlp' ,dest='dlp', choices=[DLP_TEXT, DLP_BACKEND, DLP_BACKEND_SIMPLIFIED, DLP_BACKEND_SIMPLIFIED_NCNB], help='Use the specified DLP with its apropriate encoding')

        # parse
        options, unknown = cmd_parser.parse_known_args()
        options = vars(options)

        # check
        #if options['redundancy'] and options['parallel']==0:
        #    raise Exception('command error: redundancy option must be always issued together with parallel option 1 or 2')

        # return
        return options, unknown


def run():

    # parse arguments
    options, rest = MyArgumentParser().run()

    # instance and domain
    instance = options['instance']
    if options['domain'] is not None:
        domain = options['domain']
    else:
        domain = os.path.dirname(os.path.realpath(instance)) + "/domain.pddl"
        print(domain)
        if not os.path.isfile(domain):
            domain = os.path.dirname(os.path.realpath(instance)) + "/domain_" + os.path.basename(instance)
        if not os.path.isfile(domain):
            domain = os.path.dirname(os.path.realpath(instance)) + "/../domain.pddl"

    if not os.path.isfile(domain):
        print("Domain File not found")
        return

    #
    # NORMAL CASE
    #

    # translate to facts
    if options['translate']:
        call = "{} {} {} && {} {}".format(FAST_D_TR,domain,instance,PLASP,SAS_OUTPUT)
    else:
        call = "{} {} {}".format(PLASP,domain,instance)

    # postprocess
    postprocess = ""
    if options['postprocess']:
        if options['translate'] or options['basic']:
            postprocess = " --outf=1 | grep -A1 ANSWER | tail -n1 > {} && {} {}    | clingo - {} {} && rm {}".format(TMP,PLASP,     SAS_OUTPUT,POSTPROCESS,TMP,TMP)
        else:
            postprocess = " --outf=1 | grep -A1 ANSWER | tail -n1 > {} && {} {} {} | clingo - {} {} && rm {}".format(TMP,PLASP,instance,domain,POSTPROCESS,TMP,TMP)

    # generate and test
    test = ""
    if options['test'] is not None:
        # test files
        test += "--test=- --test={} ".format(TEST_FILE)
        if options['test'] == 1:
            test += "--test={} ".format(TEST_FILE2)
        # options
        test_add = options['test_add']
        test_times = options['test_times']
        # case
        # 0: delete model
        if test_add == 0:
            test = "--test=- --test={} {}".format(TEST_FILEM, TEST_MODEL)
        # 1: forall
        elif test_add == 1:
            if test_times == 0 and not options['basic']:
                test += TEST_FORALL_1
            elif not options['basic']:
                test += TEST_FORALL_T + " --test-once"
            elif test_times == 0:
                test += TEST_B_FALL_1
            else:
                test += TEST_B_FALL_T + " --test-once"
        # 3: exists
        elif test_add == 3:
            if test_times == 0 and not options['basic']:
                test += TEST_EXISTS_1
            elif not options['basic']:
                test += TEST_EXISTS_T + " --test-once"
            elif test_times == 0:
                test += TEST_B_EX_1
            else:
                test += TEST_B_EX_T + " --test-once"
        # 4: exists with #edge
        elif test_add == 4:
            if test_times == 0 and not options['basic']:
                test += TEST_EDGE_1
            elif not options['basic']:
                test += TEST_EDGE_T + " --test-once"
            elif test_times == 0:
                test += TEST_B_EDGE_1
            else:
                test += TEST_B_EDGE_T + " --test-once"
        # 5: sequential
        elif test_add == 5:
            if test_times == 0:
                test += TEST_SEQ_1
            else:
                test += TEST_SEQ_T + " --test-once"
        # 6: dynamic sequential
        elif test_add == 6:
            if test_times == 0:
                test += TEST_DSEQ_1
            else:
                test += TEST_DSEQ_T
        # 7: delete actions
        elif test_add == 7:
            if test_times == 0:
                test += TEST_ACT_1
            else:
                test += TEST_ACT_T
        test += " "

    # heuristic
    heuristic = ""
    if options['use_heuristic']:
        heuristic = " --heuristic=Domain {} ".format(HEURISTIC)

    # preprocess
    preprocess = PREPROCESS
    if options['preprocess_simple']:
        preprocess = PRE_SIMPLE

    # normal  plan
    call += " | {} - {} {} {}".format(PLANNER,preprocess,STRIPS,
        (" ".join(rest))                                           +
        BASIC_OPTIONS                                              +
        test                                                       +
        heuristic                                                  +
        (" -c _shallow=1 " if options['shallow'] else "")          +
        " -c _closure={}  ".format(options['closure'])             +
        " -c _parallel={} ".format(options['parallel'])            +
        (" " + REDUNDANCY + " " if options['redundancy']  else "") +
        postprocess)

    #
    # OTHER CASES
    #

    # basic encoding
    if options['incmode']:
        call = call.replace(PLANNER,CLINGO + " " + INCMODE)
        call = call.replace(BASIC_OPTIONS,"")
    elif options['basic']:
        call = "{} {} {} && {} {} | {} - {} {} {} {} {}".format(
            FAST_D_TR,domain,instance,PLASP,SAS_OUTPUT,PLANNER,BASIC_OPTIONS,BASIC,test,heuristic," ".join(rest) +
               (postprocess if options['postprocess'] else "")
        )

    elif options["dlp"] is not None:
        if options["dlp"] == DLP_TEXT:
            encoding = DLP_TEXT_ENCODING 
        elif options["dlp"] == DLP_BACKEND or options["dlp"] == DLP_BACKEND_SIMPLIFIED or options["dlp"] == DLP_BACKEND_SIMPLIFIED_NCNB:
            encoding = DLP_BACKEND_ENCODING
        dlp_option = "--dlp={}".format(options["dlp"])
        parallel_options = " -c _parallel={} ".format(options['parallel'])
        shallow_option = " -c _shallow=1 " if options['shallow'] else ""
        other_options_str = " ".join([dlp_option, parallel_options, shallow_option])

        call = "{} {} {}; {} {} | {} - {} {} {} {} {}".format(
        FAST_D_TR,domain,instance,PLASP,SAS_OUTPUT,PLANNER,BASIC_OPTIONS+other_options_str,encoding,test,heuristic," ".join(rest) +
           (postprocess if options['postprocess'] else "")
    )

    # fast-downward
    elif options['fast-downward']:
        call = "{} {} {} {}".format(FAST_D,domain,instance," ".join(rest))
    # madagascar
    elif options['M']:
        call = "{} {} {} {}".format(  M,domain,instance," ".join(rest))
    elif options['Mp']:
        call = '{} {} {} {}'.format(MP,domain,instance," ".join(rest))
    elif options['MpC']:
        call = "{} {} {} {}".format(MPC,domain,instance," ".join(rest))

    #
    # SOLVE
    #

    if options['print_call'] or options['dry_run']:
        print("# planner call: " + call, file = sys.stderr)

    if options['dry_run']:
        exitCode = 0
    else:
        # the first 8 bits are reserved by the system, so the actual exit code is obtained by shifting by 8 bits
        exitCode = os.system(call) >> 8

    if options['hack']:
        os.system('echo "a." | clingo --stats -')

    # remove output.sas if it has been left over by Fast Downward
    try:
        os.remove(SAS_OUTPUT)
    except OSError:
        pass

    sys.exit(exitCode)

run()
