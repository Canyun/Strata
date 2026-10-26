// A failed conversation checkpoint must not end a streaming request (the "saving a checkpoint part failed" bug):
// host memory pressure skips the checkpoint, a dead device stays fatal, and the skips stay loud then quiet.
// Header-only rules: no GPU, no model.
#include <cstdio>

#include "strata/program/ckpt_save.hpp"

using namespace strata::program::ckpt_save;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the host ran out of memory while the GPU still answers: the request streams on without this checkpoint
    CHECK(classify(true, true) == Action::kSkip);
    // the GPU stopped responding: the reader cannot go on, whatever the save reported
    CHECK(classify(true, false) == Action::kFatal);
    CHECK(classify(false, false) == Action::kFatal);
    // a failed copy on a live device is not the GPU's fault: skip, never cancel the request
    CHECK(classify(false, true) == Action::kSkip);

    // warnings: every skip of the first ten, then one per 64 (11th is the first after the ten: 11 - 10 = 1 -> quiet
    // until the 75th: 75 - 10 = 65? no - (75-10) % 64 = 1; the 74th: 64 % 64 = 0 -> warns)
    CHECK(!warn_this_skip(0, 10, 64));
    CHECK(warn_this_skip(1, 10, 64));
    CHECK(warn_this_skip(10, 10, 64));
    CHECK(!warn_this_skip(11, 10, 64));
    CHECK(warn_this_skip(74, 10, 64));   // the 64th skip after the first ten
    CHECK(!warn_this_skip(75, 10, 64));
    CHECK(warn_this_skip(138, 10, 64));  // the 128th skip after them
    CHECK(warn_this_skip(5, 5, 0));      // every == 0: nothing warns past the first five
    CHECK(!warn_this_skip(6, 5, 0));
    CHECK(warn_this_skip(1, 1, 1));      // every == 1: every skip warns
    CHECK(warn_this_skip(1000, 1, 1));

    if (failed != 0) {
        std::fprintf(stderr, "ckpt_save_test: %d CHECK(s) failed\n", failed);
        return 1;
    }
    std::printf("ckpt_save_test: all checks passed\n");
    return 0;
}
