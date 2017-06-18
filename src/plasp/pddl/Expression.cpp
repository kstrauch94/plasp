#include <plasp/pddl/Expression.h>

#include <plasp/output/TranslatorException.h>
#include <plasp/pddl/Context.h>
#include <plasp/pddl/Domain.h>
#include <plasp/pddl/ExpressionContext.h>
#include <plasp/pddl/IO.h>
#include <plasp/pddl/expressions/And.h>
#include <plasp/pddl/expressions/Exists.h>
#include <plasp/pddl/expressions/ForAll.h>
#include <plasp/pddl/expressions/Imply.h>
#include <plasp/pddl/expressions/Not.h>
#include <plasp/pddl/expressions/Or.h>
#include <plasp/pddl/expressions/Predicate.h>
#include <plasp/pddl/expressions/PredicateDeclaration.h>
#include <plasp/pddl/expressions/Unsupported.h>
#include <plasp/pddl/expressions/When.h>

#include <tokenize/TokenizerException.h>

namespace plasp
{
namespace pddl
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Expression
//
////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::copy()
{
	return this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::normalized()
{
	return reduced()->simplified()->existentiallyQuantified()->simplified();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::reduced()
{
	return this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::existentiallyQuantified()
{
	return this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::simplified()
{
	return this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::decomposed(expressions::DerivedPredicates &)
{
	throw output::TranslatorException("expression cannot be decomposed (not normalized)");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer Expression::negated()
{
	if (is<expressions::Not>())
		return as<expressions::Not>().argument();

	auto notExpression = expressions::NotPointer(new expressions::Not);
	notExpression->setArgument(this);

	return notExpression;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: implement better (visitor pattern?)
void Expression::collectParameters(std::set<expressions::VariablePointer> &)
{
	throw output::TranslatorException("expression parameters could not be collected (expression not normalized)");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseEffectBodyExpression(Context &context, ExpressionContext &expressionContext);
ExpressionPointer parsePredicate(Context &context, ExpressionContext &expressionContext);

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parsePreconditionExpression(Context &context,
	ExpressionContext &expressionContext)
{
	auto &tokenizer = context.tokenizer;

	tokenizer.skipWhiteSpace();

	ExpressionPointer expression;

	if ((expression = expressions::And::parse(context, expressionContext, parsePreconditionExpression))
	    || (expression = expressions::ForAll::parse(context, expressionContext, parsePreconditionExpression)))
	{
		return expression;
	}

	const auto position = tokenizer.position();

	tokenizer.expect<std::string>("(");

	const auto expressionIdentifierPosition = tokenizer.position();

	if (tokenizer.testIdentifierAndSkip("preference"))
	{
		// TODO: refactor
		tokenizer.seek(expressionIdentifierPosition);
		const auto expressionIdentifier = tokenizer.getIdentifier();

		tokenizer.seek(position);
		return expressions::Unsupported::parse(context);
	}

	tokenizer.seek(position);
	return parseExpression(context, expressionContext);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseExpression(Context &context, ExpressionContext &expressionContext)
{
	auto &tokenizer = context.tokenizer;

	tokenizer.skipWhiteSpace();

	ExpressionPointer expression;

	if ((expression = expressions::And::parse(context, expressionContext, parseExpression))
	    || (expression = expressions::Or::parse(context, expressionContext, parseExpression))
	    || (expression = expressions::Exists::parse(context, expressionContext, parseExpression))
	    || (expression = expressions::ForAll::parse(context, expressionContext, parseExpression))
		|| (expression = expressions::Not::parse(context, expressionContext, parseExpression))
	    || (expression = expressions::Imply::parse(context, expressionContext, parseExpression))
	    || (expression = expressions::Predicate::parse(context, expressionContext)))
	{
		return expression;
	}

	const auto position = tokenizer.position();

	tokenizer.expect<std::string>("(");

	const auto expressionIdentifierPosition = tokenizer.position();

	if (tokenizer.testIdentifierAndSkip("-")
		|| tokenizer.testIdentifierAndSkip("=")
		|| tokenizer.testIdentifierAndSkip("*")
		|| tokenizer.testIdentifierAndSkip("+")
		|| tokenizer.testIdentifierAndSkip("-")
		|| tokenizer.testIdentifierAndSkip("/")
		|| tokenizer.testIdentifierAndSkip(">")
		|| tokenizer.testIdentifierAndSkip("<")
		|| tokenizer.testIdentifierAndSkip("=")
		|| tokenizer.testIdentifierAndSkip(">=")
		|| tokenizer.testIdentifierAndSkip("<="))
	{
		tokenizer.seek(expressionIdentifierPosition);
		const auto expressionIdentifier = tokenizer.getIdentifier();

		tokenizer.seek(position);
		return expressions::Unsupported::parse(context);
	}

	tokenizer.seek(expressionIdentifierPosition);
	const auto expressionIdentifier = tokenizer.getIdentifier();

	tokenizer.seek(position);
	throw tokenize::TokenizerException(tokenizer, "expression type “" + expressionIdentifier + "” unknown or not allowed in this context");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseEffectExpression(Context &context, ExpressionContext &expressionContext)
{
	auto &tokenizer = context.tokenizer;

	ExpressionPointer expression;

	if ((expression = expressions::And::parse(context, expressionContext, parseEffectExpression))
	    || (expression = expressions::ForAll::parse(context, expressionContext, parseEffectExpression))
	    || (expression = expressions::When::parse(context, expressionContext, parseExpression, parseConditionalEffectExpression)))
	{
		return expression;
	}

	const auto position = tokenizer.position();

	tokenizer.expect<std::string>("(");

	const auto expressionIdentifierPosition = tokenizer.position();

	if (tokenizer.testIdentifierAndSkip("when"))
	{
		tokenizer.seek(expressionIdentifierPosition);
		const auto expressionIdentifier = tokenizer.getIdentifier();

		tokenizer.seek(position);
		return expressions::Unsupported::parse(context);
	}

	tokenizer.seek(position);
	return parseEffectBodyExpression(context, expressionContext);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseEffectBodyExpression(Context &context, ExpressionContext &expressionContext)
{
	auto &tokenizer = context.tokenizer;

	ExpressionPointer expression;

	if ((expression = expressions::Not::parse(context, expressionContext, parsePredicate))
	    || (expression = expressions::Predicate::parse(context, expressionContext)))
	{
		return expression;
	}

	const auto position = tokenizer.position();

	tokenizer.expect<std::string>("(");

	const auto expressionIdentifierPosition = tokenizer.position();

	if (tokenizer.testIdentifierAndSkip("=")
		|| tokenizer.testIdentifierAndSkip("assign")
		|| tokenizer.testIdentifierAndSkip("scale-up")
		|| tokenizer.testIdentifierAndSkip("scale-down")
		|| tokenizer.testIdentifierAndSkip("increase")
		|| tokenizer.testIdentifierAndSkip("decrease"))
	{
		tokenizer.seek(expressionIdentifierPosition);
		const auto expressionIdentifier = tokenizer.getIdentifier();

		tokenizer.seek(position);
		return expressions::Unsupported::parse(context);
	}

	tokenizer.seek(expressionIdentifierPosition);
	const auto expressionIdentifier = tokenizer.getIdentifier();

	tokenizer.seek(position);
	throw tokenize::TokenizerException(tokenizer, "expression type “" + expressionIdentifier + "” unknown or not allowed in this context");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseConditionalEffectExpression(Context &context, ExpressionContext &expressionContext)
{
	ExpressionPointer expression;

	if ((expression = expressions::And::parse(context, expressionContext, parseEffectBodyExpression)))
		return expression;

	return parseEffectBodyExpression(context, expressionContext);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parsePredicate(Context &context, ExpressionContext &expressionContext)
{
	ExpressionPointer expression;

	if ((expression = expressions::Predicate::parse(context, expressionContext)))
		return expression;

	throw tokenize::TokenizerException(context.tokenizer, "expected predicate");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseLiteral(Context &context, ExpressionContext &expressionContext)
{
	ExpressionPointer expression;

	if ((expression = parseAtomicFormula(context, expressionContext))
		|| (expression = expressions::Not::parse(context, expressionContext, parseAtomicFormula)))
	{
		return expression;
	}

	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExpressionPointer parseAtomicFormula(Context &context, ExpressionContext &expressionContext)
{
	auto &tokenizer = context.tokenizer;

	ExpressionPointer expression;

	if ((expression = expressions::Predicate::parse(context, expressionContext)))
		return expression;

	const auto position = tokenizer.position();

	if (!tokenizer.testAndSkip<std::string>("("))
		return nullptr;

	const auto expressionIdentifierPosition = tokenizer.position();

	if (tokenizer.testIdentifierAndSkip("="))
	{
		tokenizer.seek(expressionIdentifierPosition);
		const auto expressionIdentifier = tokenizer.getIdentifier();

		tokenizer.seek(position);
		return expressions::Unsupported::parse(context);
	}

	tokenizer.seek(position);
	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
