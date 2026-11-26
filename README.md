# Agentic Memory for Headless Agents

Grounded prototype, adversarial invariant suite, 32-ticket evaluation harness, multi-process scale benchmarks, and Section 6 design brief solving [Agentic_Memory_Problem_Statement_IIT_BHU_2027 (1).pdf](Agentic_Memory_Problem_Statement_IIT_BHU_2027%20%281%29.pdf).

Instead of treating agent memory as a flat similarity-RAG text cache, this system implements an **epistemic belief state machine** governed by immutable server-derived trust (`SourceAuthorityRegistry`), default-deny execution (`ToolGateway`), slot-level optimistic concurrency, causal provenance DAG containment, durable refutations (`refutation_ledger`), k-anonymous two-stage global sharing, and zero-LLM critical-path recall.

---

## Measured Prototype Budgets & Invariants (Single-Host SQLite WAL)

All numbers below come directly from [benchmarks/results.json](benchmarks/results.json) (mechanism, latency, and scaling verification — **not** live LLM efficacy proof; real task-improvement claims belong to live model-in-the-loop eval via `eval/run_live_eval.py` or the 7-day production experiment in [brief/brief.md](brief/brief.md)):

| Prototype Benchmark / Metric | Budget / Target | Measured Value |
|---|---|---|
| Cumulative critical-path latency (`cold_connection_pass`, 300 runs, 2–50 steps, 8 OS workers, 62,043 mixed-status rows / 200 orgs; SQLite page cache cold, OS file cache warm) | `<= 300 ms` p95 | **`99.9 ms` p95** (`60.5 ms` p50, `47.2 ms` step-0 p95) |
| Cumulative critical-path latency (`warm_pass`, same workload) | `<= 300 ms` p95 | **`92.7 ms` p95** (`64.8 ms` p50) |
| Mixed-status recall quality (200 gold queries; `ACTIVE`/`CONTESTED`/`CANDIDATE`/`QUARANTINED`/`REFUTED`) | High relevance, `0` leaks | **`1.0` active MRR**, **`0` forbidden-status / expired-TTL leaks** |
| Max context packed per step (`PACKING_SOFT_CAP_TOKENS = 3,600`) | `<= 4,000` tokens | **`3,557` tokens** |
| Simulated 730-day workload (20 orgs, 17,544 cumulative writes, with compaction and rollups) | Growth flattens once caps apply | **`5,053` memories / `8.66 MB`** at day 730 (`8.01 MB` at day 365). This shows the caps work on this synthetic mix; it is not a forecast for a real fleet. |
| Persisted memory LLM calls per run (900 mechanism runs) | `<= 2.0` calls/run | **`0.20` calls/run** |
| Unsafe escalations in the 1,000-run mechanism check (ungated RAG baseline, for contrast) | `0` with the gateway | **Memory `0`**, Stateless `3`, ungated Naive RAG `10` |

Benchmarks are mechanism checks. Agent efficacy is reported separately in `eval/results/three_arm_offline.json`
(offline, simulated agent) and needs the live run below before any claim about real models.

---

## Review Changes (Nov 2026)

[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) has the full account of both review rounds. Summary of the second round,
in the order the review asked for:

- **Error containment.** `derived_from` (declared derivation) and `run_reads` (exposure) are now separate.
  Rejecting a report refutes the attributed root, quarantines declared descendants, keeps descendants that
  have their own CRM/human evidence, and marks claims that were only *exposed* as `CONTESTED` with
  `requires_verification` instead of quarantining them. Affected reports go to `report_review_flags`.
- **Stale memory.** Confidence (evidence quality), freshness (time since last confirmation) and
  actionability are separate. The revalidation window is the shorter of a risk window (routing 30d,
  operational 90d, diagnostic 180d) and a volatility window. Stale facts are shown to the agent as
  `STALE-REVALIDATE`, and the gateway will not act on a stale routing memory even when its domain is on the
  tenant allowlist.
- **Global evidence integrity.** Promotion checks directive polarity (negation, antonyms), version
  constraints and contributor disagreement. A synthesised advisory that drifts from the staged evidence is
  replaced by a deterministic template.
- **Attestation binding.** Seals cover tenant, run and a SHA-256 of the content, and are checked for exact equality.
- **Evaluation.** 15 tasks including silent staleness, a stale-but-still-true control and compound tickets.
  All core arms use the same gateway; optional arms isolate the gateway (ungated RAG) and staleness labels
  (`memory_no_revalidation`). Reported per arm: correct answer, unsafe action, contradiction (detected and
  unresolved), false-memory exposure and utilisation, with Wilson intervals and McNemar tests.

Offline result (simulated agent, 15 tasks, so a pilot only):

| Arm | Correct | Unsafe actions | False memory used / exposed | Avg tool calls |
|---|---|---|---|---|
| Memory | 15/15 | 0 | 0 / 11 | 3.3 |
| Stateless | 15/15 | 0 | — | 7.3 |
| Naive RAG (gated) | 13/15 | 1 | 2 / 11 | 4.1 |
| Naive RAG (ungated) | 9/15 | 5 | 6 / 11 | 3.8 |
| Memory without staleness labels | 13/15 | 0 | 2 / 11 | 3.1 |

No paired difference is significant at this size (smallest McNemar p = 0.125). The live Gemini run has not
been done in this environment; the commands are below.

---

## Repository Structure

- `src/agentic_memory/models.py` — Typed `MemoryClaim`, `CandidateClaim`, `ClaimTransition`, `PersistedCostEvent`, `TrustTier`, `RiskClass`, `MemoryStatus`, `RecallBundle`, `HumanFeedback`, and `ContainmentAuditEvent` models.
- `src/agentic_memory/store_protocol.py` — `@runtime_checkable` `MemoryStoreProtocol` decoupling epistemic policy and recall from the storage backend (`SQLiteMemoryStore` is implemented; a PostgreSQL adapter is future work).
- `src/agentic_memory/policy.py` — `SourceAuthorityRegistry` (attested `connector_id` authority seals, fail-closed `UNTRUSTED_EXTERNAL`), `TokenCounter`, `VALID_STATUS_TRANSITIONS`, `ToolCapability` default-deny `authorize_tool_action` (failing closed on untargeted external/config writes), fail-closed `infer_minimum_risk_class`, exponential volatility decay, prompt-injection detection, two-stage `sanitize_for_global`, and utility scoring.
- `src/agentic_memory/store.py` — Disk-backed SQLite WAL store with `MemoryStoreProtocol` compliance, non-hardcoded random/explicit HMAC keys (`require_explicit_hmac_secret` strict mode), `BEGIN IMMEDIATE` + re-entrant `SAVEPOINT` transactions, `Scope.GLOBAL` direct-upsert guard, atomic `promote_global_cluster` (`K >= 5` + origin/hash diversity), partial unique single-active slot index, `refutation_ledger`, `llm_cost_events`, `memory_derivations` DAG, recall-time `expires_at` filtering, FTS5 global index, and bounded rollup/pruning compaction.
- `src/agentic_memory/memory_layer.py` — `EpistemicMemoryLayer` orchestrating zero-LLM critical-path recall (relevance-first `CONTESTED`/`CANDIDATE_UNVERIFIED` ranking), structural+novelty salience gating, `derived_from_ids <= read_memory_ids` validation, run-constrained delayed feedback attribution, and `K >= 5` global promotion.
- `src/agentic_memory/llm_client.py` — `LLMClient` interface (`decide_next_action`, `extract_claims`, `adjudicate_conflict`, `attribute_feedback_root`, `synthesize_global_advisory`) with `OpenAICompatibleLLMClient`, `OllamaLLMClient`, optional experimental `GeminiLLMClient` (reads `GEMINI_API_KEY` from env), `SimulatedBlackBoxLLMClient`, and `CostLedger`.
- `src/agentic_memory/agent_runner.py` & `src/agentic_memory/experiment.py` — Unconditional `ToolGateway` boundary, blind `decide_next_action` `HeadlessAgentRunner` with encapsulated `_ToolEnvironment` (zero ground-truth peeking), and `evaluate_falsification`.
- `src/agentic_memory/extraction_policy.py` — Deterministic scope classifier (`classify_scope`), per-claim span binding, predicate/value semantic checks, and coverage-gap detection (0 LLM calls).
- `src/agentic_memory/global_semantics.py` — Directive polarity, version-constraint and contributor-conflict checks for global promotion.
- `src/agentic_memory/evaluation.py` — Outcome taxonomy (`EvaluationOutcome`), claim-level expectations, Wilson intervals, and separated extraction/admission metrics.
- `tests/test_review_hardening.py`, `tests/test_three_arm_runner.py` — regression tests for both review rounds (84 tests in the suite).
- `tests/` — the original automated tests (20 core/scenario/concurrency/edge-case tests + Scenario 3 cross-subject hard boundary + 10 adversarial invariant & 200-trial seeded property tests in `tests/test_adversarial_invariants.py` + 9 security, trust-boundary, target-key exclusion, standalone-entrypoint, rollup, blind-loop, PG16 & recall-ordering regression tests in `tests/test_security_and_boundaries.py`).
- `eval/` — 32 development tickets (`tickets_30.jsonl`), 30 held-out tickets (`tickets_heldout.jsonl`), the 15-task three-arm suite with hidden assertions (`tasks_three_arm.jsonl`), the extraction/admission harness (`run_live_eval.py`, with `as_labelled` and `mislabelled_scope` fixture variants and `--checkpoint-dir` resume), the matched three-arm runner (`run_three_arm.py`), and saved results (`results/`).
- `benchmarks/run_benchmarks.py` — Mixed-status `cold_connection_pass` / `warm_pass` multi-process latency, recall relevance (including `CONTESTED` / `CANDIDATE_UNVERIFIED`), 2-year growth trajectory across all tables, and offline blind-loop mechanism verification suite.
- `brief/` — the Section 6 submission brief ([brief/Agentic_Memory_Brief.pdf](brief/Agentic_Memory_Brief.pdf), [brief/brief.md](brief/brief.md)): six sections, ≤1,200 words, two figures (`fig1_architecture.png`, `fig2_measurements.png`). `generate_brief.py` redraws the figures from `benchmarks/results.json` and `eval/results/three_arm_offline.json`, rebuilds the PDF and checks every section limit.
- `scripts/package_submission.py` — Clean deterministic archive builder (`dist/`) that excludes local caches/databases and verifies both `pytest -q` and standalone `eval/run_live_eval.py --mode offline` from a fresh unpack.
- `ARCHITECTURE.md` — Operational constraints, explicit memory state machine, single-host SQLite vs. multi-host Postgres `MemoryStoreProtocol` migration mapping, and 11-point mechanism mapping. Only SQLite is implemented; the Postgres mapping is a design for a future adapter.

---

## Reproducing Tests, Evaluation, Benchmarks & Brief

### 1. Run Full Verification Suite (Offline Python 3.12+)
```powershell
$env:UV_LINK_MODE='copy'
uv run --python 3.12 --offline --no-project --with pydantic --with pytest --with rapidfuzz pytest -v
```
*(Or simply `pytest -v` if dependencies are installed in the active Python environment.)*

### 2. Run Extraction/Admission and Three-Arm Evaluations
Offline mock-verified mode:
```powershell
$env:UV_LINK_MODE='copy'
uv run --python 3.12 --offline --no-project --with pydantic --with rapidfuzz python eval/run_live_eval.py --mode offline
```
Live local Ollama, OpenAI, or experimental Gemini Flash endpoint (outside offline sandbox; configure keys via environment variables only):
```powershell
python eval/run_live_eval.py --mode ollama --model llama3.2
# or: python eval/run_live_eval.py --mode openai --model gpt-4o-mini
# or: $env:GEMINI_API_KEY="<key>"; python eval/run_live_eval.py --mode gemini --model gemini-3.5-flash-lite
```
Free-tier Gemini: pace requests and checkpoint so a quota stop can be resumed by rerunning the same command:
```powershell
$env:GEMINI_API_KEY="<key>"; $env:MEMORY_ATTESTATION_SECRET="<32+ hex chars>"
python eval/run_live_eval.py --mode gemini --suite extract --rpm 8 --checkpoint-dir eval/checkpoints/gemini_extract
python eval/run_three_arm.py --mode gemini --rpm 8 --checkpoint-dir eval/checkpoints/gemini_three_arm
# optional: --include-ungated-rag --include-revalidation-ablation
```

### 3. Run Mixed-Status Scale, 2-Year Growth & Mechanism Benchmarks
```powershell
$env:UV_LINK_MODE='copy'
uv run --python 3.12 --offline --no-project --with pydantic --with rapidfuzz python benchmarks/run_benchmarks.py
```

### 4. Regenerate Section 6 Design Brief PDF & Verify Word Limits
```powershell
python brief/generate_brief.py
```

### 5. Lint Check & Clean Submission Packaging
```powershell
python -m ruff check src tests benchmarks brief eval scripts
python scripts/package_submission.py
```
