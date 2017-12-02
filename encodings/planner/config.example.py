#!/usr/bin/python

import os

configuration = \
{
	"executables":
	{
		"clingo":
		{
			"command": "clingo",
		},
		"plaspTranslate":
		{
			"command": "plasp translate --parsing-mode=compatibility",
		},
		"planner":
		{
			"command": "planner.py",
			"directory": os.path.dirname(os.path.realpath(__file__)),
		},
		"fastDownward":
		{
			"command": "fast-downward.py --alias seq-sat-lama-2011 --build=release64",
			"directory": "/home/wv/bin/linux/64/fast-downward",
		},
		"fastDownwardTranslate":
		{
			"command": "fast-downward.py --translate --build=release64",
			"directory": "/home/wv/bin/linux/64/fast-downward",
		},
		"madagascarM":
		{
			"command": "M",
		},
		"madagascarMp":
		{
			"command": "Mp",
		},
		"madagascarMpC":
		{
			"command": "MpC",
		},
	},
	"encodings":
	{
		"planner":
		{
			"directory": os.path.dirname(os.path.realpath(__file__)),
			"basic": "basic.lp",
			"preprocessSimple": "preprocess_simple.lp",
			"heuristic": "heuristic.lp",
		},
		"strips":
		{
			"directory": os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "strips"),
			"preprocess": "preprocess.lp",
			"stripsIncremental": "strips-incremental.lp",
			"redundancy": "redundancy.lp",
			"postprocess": "postprocess.lp",
			"incmode": "incmode.lp",
		},
	},
	"tmpFile": os.path.join(os.path.dirname(os.path.realpath(__file__)), "run.tmp-" + str(os.getpid())),
	"basicOptions": " --query-at-last --check-at-last --forbid-actions --force-actions -c planner_on=1 ",
	"testFiles":
	{
		"directory": os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_files"),
		"test": "test.lp",
		"test_const": "test_const.lp",
		"test_model": "test_model.lp",
		"block_model": "block_model.lp",
		"block_forall_1": "block_forall_1.lp",
		"block_forall_t": "block_forall_t.lp",
		"block_forall_1_basic": "block_forall_1_basic.lp",
		"block_forall_t_basic": "block_forall_t_basic.lp",
		"block_exists_1": "block_exists_1.lp",
		"block_exists_t": "block_exists_t.lp",
		"block_exists_1_basic": "block_exists_1_basic.lp",
		"block_exists_t_basic": "block_exists_t_basic.lp",
		"block_exists_edge_1": "block_exists_edge_1.lp",
		"block_exists_edge_t": "block_exists_edge_t.lp",
		"block_exists_edge_1_basic": "block_exists_edge_1_basic.lp",
		"block_exists_edge_t_basic": "block_exists_edge_t_basic.lp",
		"block_sequential_1": "block_sequential_1.lp",
		"block_sequential_t": "block_sequential_t.lp",
		"block_dyn_sequential_1": "block_dyn_sequential_1.lp",
		"block_dyn_sequential_t": "block_dyn_sequential_t.lp",
		"block_actions_1": "block_actions_1.lp",
		"block_actions_t": "block_actions_t.lp",
	},
}
