#include <catch.hpp>

#include <experimental/filesystem>

#include <pddlparse/AST.h>
#include <pddlparse/Parse.h>

namespace fs = std::experimental::filesystem;

const pddl::Context::WarningCallback ignoreWarnings = [](const auto &, const auto &){};
const auto pddlInstanceBasePath = fs::path("data") / "pddl-instances";

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("[PDDL parser] The official PDDL instances are parsed correctly", "[PDDL parser]")
{
	pddl::Tokenizer tokenizer;
	pddl::Context context(std::move(tokenizer), ignoreWarnings);

	SECTION("types, predicates, and actions in blocksworld domain")
	{
		const auto domainFile = pddlInstanceBasePath / "ipc-2000" / "domains" / "blocks-strips-typed" / "domain.pddl";
		context.tokenizer.read(domainFile);
		auto description = pddl::parseDescription(context);

		CHECK(description.domain->name == "blocks");
		CHECK(description.domain->constants.empty());

		const auto &types = description.domain->types;

		REQUIRE(types.size() == 1);
		const auto &typeBlock = types[0];
		CHECK(typeBlock->name == "block");

		const auto &predicates = description.domain->predicates;

		REQUIRE(predicates.size() == 5);

		CHECK(predicates[0]->name == "on");
		REQUIRE(predicates[0]->parameters.size() == 2);
		CHECK(predicates[0]->parameters[0]->name == "x");
		CHECK(predicates[0]->parameters[0]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(predicates[0]->parameters[1]->name == "y");
		CHECK(predicates[0]->parameters[1]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(predicates[1]->name == "ontable");
		REQUIRE(predicates[1]->parameters.size() == 1);
		CHECK(predicates[1]->parameters[0]->name == "x");
		CHECK(predicates[1]->parameters[0]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(predicates[2]->name == "clear");
		REQUIRE(predicates[2]->parameters.size() == 1);
		CHECK(predicates[2]->parameters[0]->name == "x");
		CHECK(predicates[2]->parameters[0]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(predicates[3]->name == "handempty");
		CHECK(predicates[3]->parameters.empty());
		CHECK(predicates[4]->name == "holding");
		REQUIRE(predicates[4]->parameters.size() == 1);
		CHECK(predicates[4]->parameters[0]->name == "x");
		CHECK(predicates[4]->parameters[0]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());

		const auto &actions = description.domain->actions;

		REQUIRE(actions.size() == 4);

		CHECK(actions[3]->name == "unstack");

		REQUIRE(actions[3]->parameters.size() == 2);
		CHECK(actions[3]->parameters[0]->name == "x");
		CHECK(actions[3]->parameters[0]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(actions[3]->parameters[1]->name == "y");
		CHECK(actions[3]->parameters[1]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());

		const auto &preconditionAnd = actions[3]->precondition.value().get<pddl::ast::AndPointer<pddl::ast::Precondition>>();
		const auto &precondition0 = preconditionAnd->arguments[0].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(precondition0->arguments.size() == 2);
		CHECK(precondition0->arguments[0].get<pddl::ast::VariablePointer>()->declaration->name == "x");
		CHECK(precondition0->arguments[0].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(precondition0->arguments[1].get<pddl::ast::VariablePointer>()->declaration->name == "y");
		CHECK(precondition0->arguments[1].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &precondition1 = preconditionAnd->arguments[1].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(precondition1->arguments.size() == 1);
		CHECK(precondition1->arguments[0].get<pddl::ast::VariablePointer>()->declaration->name == "x");
		CHECK(precondition1->arguments[0].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &precondition2 = preconditionAnd->arguments[2].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(precondition2->arguments.empty());

		const auto &effectAnd = actions[3]->effect.value().get<pddl::ast::AndPointer<pddl::ast::Effect>>();
		const auto &effect0 = effectAnd->arguments[0].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(effect0->arguments.size() == 1);
		CHECK(effect0->arguments[0].get<pddl::ast::VariablePointer>()->declaration->name == "x");
		CHECK(effect0->arguments[0].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &effect1 = effectAnd->arguments[1].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(effect1->arguments.size() == 1);
		CHECK(effect1->arguments[0].get<pddl::ast::VariablePointer>()->declaration->name == "y");
		CHECK(effect1->arguments[0].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &effectNot2 = effectAnd->arguments[2].get<pddl::ast::NotPointer<pddl::ast::Effect>>()->argument.get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(effectNot2->arguments.size() == 1);
		CHECK(effectNot2->arguments[0].get<pddl::ast::VariablePointer>()->declaration->name == "x");
		CHECK(effectNot2->arguments[0].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &effectNot3 = effectAnd->arguments[3].get<pddl::ast::NotPointer<pddl::ast::Effect>>()->argument.get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(effectNot3->arguments.empty());
		const auto &effectNot4 = effectAnd->arguments[4].get<pddl::ast::NotPointer<pddl::ast::Effect>>()->argument.get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(effectNot4->arguments.size() == 2);
		CHECK(effectNot4->arguments[0].get<pddl::ast::VariablePointer>()->declaration->name == "x");
		CHECK(effectNot4->arguments[0].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(effectNot4->arguments[1].get<pddl::ast::VariablePointer>()->declaration->name == "y");
		CHECK(effectNot4->arguments[1].get<pddl::ast::VariablePointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
	}

	SECTION("types, predicates, and actions in blocksworld instance")
	{
		const auto domainFile = pddlInstanceBasePath / "ipc-2000" / "domains" / "blocks-strips-typed" / "domain.pddl";
		const auto instanceFile = pddlInstanceBasePath / "ipc-2000" / "domains" / "blocks-strips-typed" / "instances" / "instance-1.pddl";
		context.tokenizer.read(domainFile);
		context.tokenizer.read(instanceFile);
		auto description = pddl::parseDescription(context);

		const auto &types = description.domain->types;
		const auto &typeBlock = types[0];

		REQUIRE(description.problem);

		const auto &problem = description.problem.value();

		CHECK(problem->name == "blocks-4-0");
		CHECK(problem->domain->name == "blocks");

		const auto &objects = problem->objects;

		REQUIRE(objects.size() == 4);
		CHECK(objects[0]->name == "d");
		CHECK(objects[1]->name == "b");
		CHECK(objects[2]->name == "a");
		CHECK(objects[3]->name == "c");

		const auto &facts = problem->initialState.facts;

		REQUIRE(facts.size() == 9);
		const auto &fact0 = facts[0].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(fact0->arguments.size() == 1);
		CHECK(fact0->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->name == "c");
		CHECK(fact0->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &fact5 = facts[5].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(fact5->arguments.size() == 1);
		CHECK(fact5->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->name == "a");
		CHECK(fact5->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &fact8 = facts[8].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(fact8->arguments.empty());

		REQUIRE(problem->goal);

		const auto &goal = problem->goal.value();
		const auto &goalAnd = goal.get<pddl::ast::AndPointer<pddl::ast::Precondition>>();

		REQUIRE(goalAnd->arguments.size() == 3);
		const auto &goal0 = goalAnd->arguments[0].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(goal0->arguments.size() == 2);
		CHECK(goal0->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->name == "d");
		CHECK(goal0->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(goal0->arguments[1].get<pddl::ast::ConstantPointer>()->declaration->name == "c");
		CHECK(goal0->arguments[1].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &goal1 = goalAnd->arguments[1].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(goal0->arguments.size() == 2);
		CHECK(goal1->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->name == "c");
		CHECK(goal1->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(goal1->arguments[1].get<pddl::ast::ConstantPointer>()->declaration->name == "b");
		CHECK(goal1->arguments[1].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		const auto &goal2 = goalAnd->arguments[2].get<pddl::ast::AtomicFormula>().get<pddl::ast::PredicatePointer>();
		// TODO: check declaration once implemented
		REQUIRE(goal0->arguments.size() == 2);
		CHECK(goal2->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->name == "b");
		CHECK(goal2->arguments[0].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
		CHECK(goal2->arguments[1].get<pddl::ast::ConstantPointer>()->declaration->name == "a");
		CHECK(goal2->arguments[1].get<pddl::ast::ConstantPointer>()->declaration->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration == typeBlock.get());
	}

	SECTION("“either” type in zenotravel domain")
	{
		const auto domainFile = pddlInstanceBasePath / "ipc-2002" / "domains" / "zenotravel-numeric-hand-coded" / "domain.pddl";
		context.tokenizer.read(domainFile);
		auto description = pddl::parseDescription(context);

		const auto &predicates = description.domain->predicates;

		REQUIRE(predicates.size() == 2);
		REQUIRE(predicates[0]->name == "at");
		REQUIRE(predicates[0]->parameters.size() == 2);
		REQUIRE(predicates[0]->parameters[0]->name == "x");
		REQUIRE(predicates[0]->parameters[0]->type);
		CHECK(predicates[0]->parameters[0]->type.value().get<pddl::ast::EitherPointer<pddl::ast::PrimitiveTypePointer>>()->arguments[0]->declaration->name == "person");
		CHECK(predicates[0]->parameters[0]->type.value().get<pddl::ast::EitherPointer<pddl::ast::PrimitiveTypePointer>>()->arguments[1]->declaration->name == "aircraft");
		REQUIRE(predicates[0]->parameters[1]->name == "c");
		REQUIRE(predicates[0]->parameters[1]->type);
		CHECK(predicates[0]->parameters[1]->type.value().get<pddl::ast::PrimitiveTypePointer>()->declaration->name == "city");
	}
}
