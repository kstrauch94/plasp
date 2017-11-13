#ifndef __TOKENIZE__LOCATION_H
#define __TOKENIZE__LOCATION_H

#include <string>

#include <tokenize/StreamPosition.h>

namespace tokenize
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Location
//
////////////////////////////////////////////////////////////////////////////////////////////////////

class Stream;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Location
{
	StreamPosition position{InvalidStreamPosition};

	// TODO: think about avoiding copying strings
	std::string sectionStart;
	std::string sectionEnd;

	StreamPosition rowStart{InvalidStreamPosition};
	StreamPosition rowEnd{InvalidStreamPosition};

	StreamPosition columnStart{InvalidStreamPosition};
	StreamPosition columnEnd{InvalidStreamPosition};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}

#endif
