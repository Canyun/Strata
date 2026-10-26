// include/strata/program/ckpt_save.hpp - the rules that keep a failed conversation checkpoint from ending a request.
//
// A checkpoint save copies the session's running state into host vectors: under memory pressure the resize throws.
// #752 set the rule for parking ("a host OOM skips parking instead of ending the server"); the prompt path's saves
// did not follow it - a failed save cancelled the request, and the server showed "LLM streaming error: saving a
// checkpoint part failed".  These rules draw the line the prompt path needs: a save that only ran out of memory
// skips the checkpoint and the request streams on; a device that stopped responding stays fatal and the reader
// must still cancel; the skip is loud for the first warnings and quiet afterwards, so a full host is visible in
// the log without one line per chunk.
#pragma once

#include <cstdint>

namespace strata::program::ckpt_save {

/// What a failed checkpoint save means for the request in flight.
enum class Action {
    kSkip,  // the state is fine, the checkpoint simply was not written: the request streams on
    kFatal  // the device itself is the failure: the prompt reader must cancel
};

/// A save that ran out of memory while the device still answers is a shortage of host memory - the copy is
/// host-side - so it is a skip.  A device that stopped responding is fatal, allocation failure or not.
inline Action classify(bool allocation_failed, bool device_responding) {
    if (!device_responding) return Action::kFatal;
    if (allocation_failed) return Action::kSkip;
    return Action::kSkip;   // any other copy failure with a live device: the checkpoint is optional, skip it
}

/// The warning throttle: the first `first` skips warn, every `every`-th after that does.
/// `skip_count` is the number of skips so far, counted from 1 (the first skip passes 1).
inline bool warn_this_skip(uint64_t skip_count, uint64_t first, uint64_t every) {
    if (skip_count == 0) return false;
    if (skip_count <= first) return true;
    if (every == 0) return false;
    return (skip_count - first) % every == 0;
}

/// Defaults a serving process uses: ten warnings, then one per 64 skips.
constexpr uint64_t kWarnFirst = 10;
constexpr uint64_t kWarnEvery = 64;

}  // namespace strata::program::ckpt_save
