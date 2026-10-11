// include/strata/core/deterministic.hpp - STRATA_DETERMINISTIC=1 (opt-in): the engine's run-to-run switch.
//
// Unset (the default) nothing changes.  Set to 1, every decision that can change a result and is taken from the clock or
// from a measurement is replaced by a fixed rule, so two fresh engines of the same build, flags and prompts give the same
// token ids at temperature 0.  What it pins and what it cannot is in docs/DETAILS.md ("Run-to-run determinism").
#pragma once

#include <cstdlib>

namespace strata::core {

inline bool deterministic_mode() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DETERMINISTIC");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

}  // namespace strata::core
