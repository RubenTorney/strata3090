// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap.
#include "strata/spec/draft_policy.hpp"

#include <cstdio>
#include <limits>

using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        // #1252: a wide size priced high in one slow stretch is measured again after a while
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 4; ++i) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 6; ++i) p.observe(true, 6, 5, 40, 4.0 * cost(6));  // the wide size, in a slow stretch
        for (int i = 0; i < 30; ++i) p.observe(true, 4, 3, 40, cost(4));       // and the short window's rate is fine
        bool saw6 = false;
        for (int i = 0; i < 400 && !saw6; ++i) {
            const DraftPolicy::Pick k = p.choose(4, 5, 40);
            if (k.lookup && k.t == 6) { saw6 = true; p.observe(true, 6, 5, 40, cost(6)); }
            else p.observe(k.lookup, k.t, k.t - 1, 40, cost(k.t));
        }
        check(saw6, "a stale wide size is tried again within a few hundred rounds");
    }
    {
        // --lookup-chain: rows cheap (UMA-like) and the chain always accepted -> chained; rows dear or the chain
        // always rejected -> not
        auto cheap = [](int t) { return 40.0 + 2.0 * (t - 1); };
        DraftPolicy p(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) p.observe(false, t, 1, 0, cheap(t));
        for (int i = 0; i < 30; ++i) p.observe(false, 4, 2, 0, cheap(4));
        for (int i = 0; i < 30; ++i) p.observe_chain(4, 3, 6, 12, cheap(7));    // every chained token accepted
        check(p.chain(4, 0.9, 3, 12) == 3, "chain: cheap rows, always accepted -> all 3 chained");
        DraftPolicy q(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) q.observe(false, t, 1, 0, cost(t));
        for (int i = 0; i < 30; ++i) q.observe_chain(4, 3, 3, 4, cost(7));       // reached, never accepted
        check(q.chain(4, 0.9, 3, 4) == 0, "chain: dear rows, never accepted -> none");
        check(q.chain(4, 0.9, 0, 4) == 0 && q.chain(8, 1.0, 3, 40) == 0, "chain: nothing proposed / no room -> none");
    }
    {
        // The stale-size check must also run when a shorter chain already pays. Synthetic costs isolate the
        // selection policy: even a size that stays expensive is retried, but only once per stale interval.
        DraftPolicy p(6);
        for (int i = 0; i < 3; ++i) p.observe(false, 4, 3, 0, 40.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 1, 4, 40, 42.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 2, 5, 40, 1000.0);
        check(p.chain(4, 1.0, 2, 40) == 1, "chain: the profitable shorter chain initially wins");
        int probes = 0, last_probe = -1;
        bool spaced = true, confident = true;
        for (int i = 0; i < 1000; ++i) {
            const int k = p.chain(4, 1.0, 2, 40);
            if (k == 2) {
                ++probes;
                spaced &= last_probe < 0 ? i >= 300 : i - last_probe > 300;
                last_probe = i;
                confident &= p.chain(4, 0.5, 2, 40) < 2;
            }
            p.observe_chain(4, k, 3 + k, 40, k == 2 ? 1000.0 : 42.0);
        }
        check(probes > 0 && probes <= 3 && spaced,
              "chain: stale full size is re-probed despite a profitable shorter chain");
        check(probes > 0 && confident, "chain: stale re-probes retain the MTP confidence gate");
    }
    {   // mtp_window_len: the per-guess rule (--spec-min-p) against the cumulative one (--spec-cum-p)
        const float pr[5] = {0.9f, 0.8f, 0.75f, 0.72f, 0.95f};
        using strata::spec::mtp_window_len;
        check(mtp_window_len(pr, 6, 0.0, 0.0) == 6, "window: no rule takes the cap");
        check(mtp_window_len(pr, 6, 0.7, 0.0) == 6, "window: per-guess 0.7 keeps every guess here (each is >= 0.7)");
        check(mtp_window_len(pr, 6, 0.7, 0.5) == 4, "window: cumulative 0.5 (0.9, 0.72, 0.54, 0.39) keeps 3 guesses");
        check(mtp_window_len(pr, 6, 0.0, 0.3) == 6, "window: cumulative 0.3 keeps all five guesses");
        check(mtp_window_len(pr, 3, 0.0, 0.1) == 3, "window: the cap holds");
        const float zero[2] = {0.0f, 0.0f};
        check(mtp_window_len(zero, 3, 0.0, 0.1) == 1, "window: guesses never drafted (probability 0) are left out");
    }
    {   // the old inline rule (--spec-min-p alone), the NaN cases, the stop order, and the drafter's stop loop
        using strata::spec::mtp_window_len;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        auto old_rule = [](const float* d, int S, double mp) {
            int T = 1;
            while (T < S && d[T - 1] >= (float) mp) ++T;
            return T;
        };
        // the drafter's loop in MtpDrafter::draft (min_p > 0 or cum_p > 0): how many drafts it computes
        auto drafted = [](const float* d, int max_steps, float min_p, float cum_p) {
            int n = 1;
            float pj = d[0], cum = pj;
            for (int j = 1; j < max_steps && pj >= min_p && cum >= cum_p; ++j) {
                pj = d[j];
                cum *= pj;
                ++n;
            }
            return n;
        };
        const float edge[5] = {(float) 0.7, (float) 0.7, 0.69999f, 0.9f, 0.9f};
        check(mtp_window_len(edge, 6, 0.7, 0.0) == old_rule(edge, 6, 0.7), "parity: dprob == (float)0.7 enters, 0.69999 stops");
        check(mtp_window_len(edge, 6, 0.7, 0.0) == 3, "parity: that window is 3");
        bool same = true;
        const float vals[] = {0.0f, 0.3f, 0.5f, 0.69999f, (float) 0.7, 0.8f, 0.95f, 1.0f, nan};
        for (float a : vals) for (float b : vals) for (float c : vals) {
            const float d[3] = {a, b, c};
            for (double mp : {0.0, 0.5, 0.7})
                if (mp > 0.0 && mtp_window_len(d, 4, mp, 0.0) != old_rule(d, 4, mp)) same = false;
        }
        check(same, "parity: cum_p = 0 equals the old inline rule on a grid incl. NaN");
        const float bad[3] = {0.9f, nan, 0.9f};
        check(mtp_window_len(bad, 4, 0.7, 0.0) == 2, "NaN: --spec-min-p stops the window before it");
        check(mtp_window_len(bad, 4, 0.0, 0.3) == 2, "NaN: cum-p alone stops the window before it");
        const float lowq[3] = {0.6f, 0.99f, 0.99f};
        check(mtp_window_len(lowq, 4, 0.7, 0.1) == 1, "min-p stops before cum-p (0.6 < 0.7, product 0.6 >= 0.1)");
        const float lowc[3] = {0.8f, 0.8f, 0.8f};
        check(mtp_window_len(lowc, 4, 0.7, 0.7) == 2, "cum-p stops what min-p keeps (0.8, 0.64)");
        check(mtp_window_len(lowc, 1, 0.7, 0.7) == 1, "max_t = 1: a window of one");
        check(mtp_window_len(lowc, 4, 0.0, 1.0) == 1, "P = 1.0: windows of one (0.8 < 1)");
        bool atleast = true;
        for (float a : vals) for (float b : vals) for (float c : vals) {
            const float d[3] = {a, b, c};
            for (double mp : {0.0, 0.7}) for (double cp : {0.0, 0.3, 0.5, 0.9}) {
                if (mp <= 0.0 && cp <= 0.0) continue;
                if (drafted(d, 3, (float) mp, (float) cp) < mtp_window_len(d, 4, mp, cp) - 1) atleast = false;
            }
        }
        check(atleast, "the drafter's stop loop drafts at least T-1 guesses for every window");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
