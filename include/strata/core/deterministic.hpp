// include/strata/core/deterministic.hpp - STRATA_DETERMINISTIC=1 (opt-in): the engine's run-to-run switch.
//
// Unset (the default) nothing changes.  Set to 1 (or true, yes, on), every decision that can change a result and is taken from
// the clock or from a measurement is replaced by a fixed rule, so two fresh engines of the same build, flags and prompts are
// meant to give the same token ids at temperature 0 (not yet shown on an engine).  What it pins and what it cannot is in
// docs/DETAILS.md ("Run-to-run determinism").
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

inline bool deterministic_mode() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DETERMINISTIC");
        if (v == nullptr || v[0] == '\0') return false;
        auto is = [&](const char* w) {
            size_t i = 0;
            for (; w[i] != '\0'; ++i)
                if (v[i] == '\0' || (v[i] | 0x20) != w[i]) return false;
            return v[i] == '\0';
        };
        if (is("1") || is("true") || is("yes") || is("on")) return true;
        std::fprintf(stderr, "strata: STRATA_DETERMINISTIC=%s is not 1/true/yes/on: the deterministic mode is OFF\n", v);
        return false;
    }();
    return on;
}

}  // namespace strata::core
