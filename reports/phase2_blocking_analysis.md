# Phase 2 Blocking Analysis

## Development Sample

- S1 sample size: 10,000 (seed 42)
- S1 rows with ground-truth matches: 9,462
- Total true matched IDs in sample: 34,723

Ground truth was used only to score candidates after generation, never to construct blocking keys.

## Normalization fixes carried into this run

- Regex escaping was corrected so the normalization pattern is interpreted as intended by the regex engine, rather than matching literal backslash characters.
- Character classes switched from ASCII `\w`/`\s` to Unicode `\p{L}`/`\p{N}`, since DuckDB's RE2 engine treats `\w` as ASCII-only. Without this, every non-Latin-script business name (Devanagari, Tamil, Gujarati, etc.) and every accented Latin name normalized to an empty or broken string.

## Individual Blocking Passes

| Rule | Candidate pairs | Recall | Avg/S1 | Max/S1 | Zero-cand. S1 |
|---|---:|---:|---:|---:|---:|
| B1 | 95,857 | 22.2936% | 13.31 | 402 | 2,796 |
| B2 | 1,249,961 | 36.4686% | 245.19 | 1,610 | 4,902 |
| B3 | 3,682 | 7.7240% | 2.82 | 58 | 8,696 |
| B4 | 682,063 | 33.2690% | 97.58 | 8,170 | 3,010 |
| B5 | 408,268 | 42.4531% | 44.34 | 1,495 | 793 |
| B6 | 967,367 | 33.9660% | 191.63 | 1,583 | 4,952 |
| B7 | 1,847,715 | 49.3247% | 272.40 | 1,978 | 3,217 |

- **B1** (country + exact normalized business name): Exact match; highest precision, catches identical naming only.
- **B2** (country + rarest normalized name token (freq-capped)): Each entity's single rarest token (len>=4, corpus freq<=2,000).
- **B3** (country + address number + SECOND-rarest name token (independent of B2)): Uses each entity's SECOND-rarest token (rn=2), not the one B2 already used, combined with extracted address number, so it is independent of B2 rather than a subset.
- **B4** (country + address signature (first 3 address tokens)): Compact address signature; tolerant of minor address wording differences.
- **B5** (country + suffix-stripped exact name (legal-suffix insensitive)): Strips legal-entity suffix words (llc, inc, ltd, ltda, pvt, private, ...) before exact match.
- **B6** (country + order-independent two-token signature): Each entity's two rarest qualifying tokens, sorted alphabetically and joined -- order-independent and tighter than B2 alone, at the cost of requiring 2 qualifying tokens.
- **B7** (country + frequency-capped 6-char name prefix): First 6 chars of suffix-stripped, space-removed name; prefixes with corpus frequency > 2,000 are excluded to bound volume.

## Cumulative Union (fixed evaluation order, with incremental gain)

| Step | Candidate pairs | Recall | Avg/S1 | New true IDs from this rule |
|---|---:|---:|---:|---:|
| +B1 (B1) | 95,857 | 22.2936% | 13.31 | 7,741 |
| +B5 (B1+B5) | 408,268 | 42.4531% | 44.34 | 7,000 |
| +B2 (B1+B5+B2) | 1,616,622 | 55.5511% | 166.28 | 4,548 |
| +B3 (B1+B5+B2+B3) | 1,617,740 | 56.0349% | 166.40 | 168 |
| +B4 (B1+B5+B2+B3+B4) | 2,293,505 | 71.6096% | 231.71 | 5,408 |
| +B6 (B1+B5+B2+B3+B4+B6) | 2,293,505 | 71.6096% | 231.71 | 0 |
| +B7 (B1+B5+B2+B3+B4+B6+B7) | 3,685,634 | 79.7541% | 370.30 | 2,828 |

## Rule Inclusion Rationale

- **B1 kept**: added 7,741 true IDs not already found by (nothing).
- **B5 kept**: added 7,000 true IDs not already found by B1.
- **B2 kept**: added 4,548 true IDs not already found by B1+B5.
- **B3 kept**: added 168 true IDs not already found by B1+B5+B2.
- **B4 kept**: added 5,408 true IDs not already found by B1+B5+B2+B3.
- **B6 contributed no incremental recall** in this evaluation order (0 or negative new true IDs on top of B1+B5+B2+B3+B4). Kept in the benchmark for transparency, but excluded from the recommended configuration unless reordering changes this.
- **B7 kept**: added 2,828 true IDs not already found by B1+B5+B2+B3+B4+B6.

## Recommended Production Configuration

**B1+B5+B2+B3+B4+B6+B7** -- selected as the cumulative step with the highest measured recall (79.7541%) while keeping average candidates/S1 (370.30) at or under the configured ceiling of 500.

## Limitations

- Measured recall tops out at 79.7541% for the full union of all seven passes on this development sample. If your project target is materially higher, these deterministic exact/near-exact keys are not sufficient on their own -- true matches with heavier typos, transliteration differences, or unrelated wording will not share any of these keys and need a genuinely fuzzy pass (e.g. phonetic encoding, edit-distance-bounded blocking) on top of this set.
- All measurements are on a 10,000-row S1 development sample; recall and volume should be re-validated on a larger sample before being treated as final production numbers.
- B3's redesign only uses each entity's second-rarest qualifying token; entities with fewer than two qualifying tokens (short or highly generic names) get no B3 candidates and rely on B1/B2/B4/B5/B6/B7 instead.
