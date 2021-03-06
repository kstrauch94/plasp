#include <plasp/Language.h>

#include <map>

namespace plasp
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Language
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static const std::map<std::string, Language::Type> languageNames =
	{
		{"auto", Language::Type::Automatic},
		{"pddl", Language::Type::PDDL},
		{"sas", Language::Type::SAS},
	};

////////////////////////////////////////////////////////////////////////////////////////////////////

Language::Type Language::fromString(const std::string &languageName)
{
	const auto matchingLanguageName = languageNames.find(languageName);

	if (matchingLanguageName == languageNames.cend())
		return Language::Type::Unknown;

	return matchingLanguageName->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}
