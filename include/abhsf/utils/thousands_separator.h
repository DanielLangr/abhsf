#ifndef UTILS_THOUSANDS_SEPARATOR_H
#define UTILS_THOUSANDS_SEPARATOR_H

#include <locale>
#include <string>

struct thousands_separator : std::numpunct<char>
{
    char do_thousands_sep() const { return ','; }
    std::string do_grouping() const { return "\3"; }
};

#endif
