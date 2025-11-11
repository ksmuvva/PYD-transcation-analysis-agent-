# Session Memory Requirements Analysis

## Executive Summary

This document analyzes the 23 session memory requirements (REQ-SM-001 through REQ-SM-023) for applicability to the **batch processing pipeline architecture** of the transaction analysis agent.

**Architecture Type**: Batch Pipeline Processor (not conversational agent)
**Current State**: Primitive session memory via `AgentDependencies.results` dict
**Target**: Formal session memory with SessionContext, MCTS caching, and comprehensive observability

---

## Requirements Assessment

### ✅ APPLICABLE & REQUIRED (13 Requirements)

| Requirement | Status | Priority | Notes |
|------------|--------|----------|-------|
| REQ-SM-001 | Partially Implemented | HIGH | Session boundary implicit, needs explicit UUID |
| REQ-SM-002 | Implemented | HIGH | Already uses `deps_type=AgentDependencies` |
| REQ-SM-003 | **NOT IMPLEMENTED** | **CRITICAL** | Need SessionContext Pydantic model |
| REQ-SM-004 | Implemented | MEDIUM | Pipeline already sequential |
| REQ-SM-005 | Implemented | MEDIUM | Tool 1 writes to deps.results |
| REQ-SM-006 | Implemented | MEDIUM | Tool 2 reads Tool 1 output |
| REQ-SM-007 | Implemented | MEDIUM | Tool 3 reads Tools 1 & 2 output |
| REQ-SM-008 | Implemented | MEDIUM | Tool 4 aggregates all results |
| REQ-SM-009 | **NOT IMPLEMENTED** | **HIGH** | Need UUID generation per CSV |
| REQ-SM-013 | **NOT IMPLEMENTED** | **HIGH** | MCTS cache = major cost savings |
| REQ-SM-014 | **NOT IMPLEMENTED** | HIGH | Cache invalidation logic |
| REQ-SM-015 | **NOT IMPLEMENTED** | HIGH | Cache isolation between sessions |
| REQ-SM-016 | Partially Implemented | HIGH | Root span exists, needs session_id |

### ⚠️ APPLICABLE & NEEDS ENHANCEMENT (7 Requirements)

| Requirement | Current Gap | Priority | Implementation Effort |
|------------|-------------|----------|----------------------|
| REQ-SM-010 | No explicit memory cleanup | MEDIUM | Low (add gc.collect() + logging) |
| REQ-SM-017 | No session_id in spans | HIGH | Low (add attribute to all spans) |
| REQ-SM-018 | No session metrics | MEDIUM | Medium (aggregate costs/timing) |
| REQ-SM-019 | No completion event | MEDIUM | Low (add final Logfire span) |
| REQ-SM-020 | No debug mode | LOW | Medium (optional feature) |
| REQ-SM-021 | Already implemented | N/A | Tools use RunContext correctly |
| REQ-SM-022 | Mutates dict directly | HIGH | Low (use model_copy pattern) |

### ❌ NOT APPLICABLE (3 Requirements)

| Requirement | Reason | Alternative |
|------------|--------|-------------|
| REQ-SM-011 | No concurrent execution in batch processor | Could implement for future API mode |
| REQ-SM-012 | No long-running sessions (runs complete in minutes) | N/A - batch jobs don't idle |
| REQ-SM-023 | Testing requirement | Will implement as separate task |

---

## Architecture Context

### Current Design: Batch Pipeline Processor

```
CSV Upload (Session Start)
    ↓
1. Filter Transactions (Tool 1)
    ↓ [writes to deps.results['filtered_df']]
2. Classify Each Transaction (Tool 2)
    ↓ [reads filtered_df, writes classifications]
3. Detect Fraud Each Transaction (Tool 3)
    ↓ [reads filtered_df + classifications]
4. Generate Enhanced CSV (Tool 4)
    ↓ [reads all prior results]
CSV Output (Session End) + Memory Cleanup
```

**Key Characteristics:**
- Single-pass execution (no multi-turn dialogue)
- Tools called directly via `RunContext`, not through `agent.run()`
- Session lifetime: ~2-10 minutes (depends on CSV size)
- No user interaction during processing
- No state retention between CSV uploads

### Why This Matters for Session Memory

**Conversational Agent vs Batch Processor:**

| Feature | Conversational Agent | This Batch Processor |
|---------|---------------------|---------------------|
| Session Duration | Hours/days (multi-turn) | Minutes (single-run) |
| User Interaction | Continuous dialogue | None (fire-and-forget) |
| State Retention | Cross-conversation memory | Cross-tool memory only |
| Timeout Handling | Yes (idle sessions) | No (always active) |
| Concurrent Sessions | Yes (multi-user) | No (single CSV at a time) |
| Session Cleanup | On timeout/logout | On pipeline completion |

**Implication**: Requirements designed for conversational agents (timeouts, concurrent sessions) don't apply here.

---

## Implementation Plan

### Phase 1: Core Session Memory (CRITICAL)

**Goal**: Replace primitive dict-based memory with formal SessionContext

```python
# NEW: SessionContext Pydantic model
class SessionContext(BaseModel):
    # Session identity
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    csv_file_name: str
    session_start_time: float = Field(default_factory=time.time)

    # Pipeline data
    raw_csv_data: pd.DataFrame
    filtered_transactions: pd.DataFrame | None = None
    classification_results: dict[str, ClassificationResult] = Field(default_factory=dict)
    fraud_results: dict[str, FraudDetectionResult] = Field(default_factory=dict)

    # MCTS optimization
    mcts_cache: dict[str, MCTSCacheEntry] = Field(default_factory=dict)
    mcts_cache_hits: int = 0
    mcts_cache_misses: int = 0

    # Execution tracking
    execution_log: list[dict[str, Any]] = Field(default_factory=list)

    # Final output
    output_csv_path: str | None = None

    # Config (immutable per session)
    config: AgentConfig
    mcts_engine: MCTSEngine
    llm_client: Model

    class Config:
        arbitrary_types_allowed = True  # For DataFrame, MCTSEngine, Model
```

**Tasks:**
- [x] Design SessionContext schema (REQ-SM-003)
- [ ] Create `src/session_context.py` with Pydantic model
- [ ] Update agent initialization to use `deps_type=SessionContext`
- [ ] Migrate `AgentDependencies` references to `SessionContext`
- [ ] Add session_id generation (REQ-SM-009)

### Phase 2: MCTS Caching (HIGH VALUE)

**Goal**: Cache MCTS trees for similar transactions within same CSV

**Cache Key Strategy:**
```python
def compute_cache_key(transaction: dict) -> str:
    merchant_category = transaction.get('merchant_category', 'unknown')
    amount_gbp = transaction['amount_gbp']
    # Round to nearest £5 for fuzzy matching
    amount_bucket = round(amount_gbp / 5) * 5
    return f"{merchant_category}_{amount_bucket}"

# Usage
cache_key = compute_cache_key(transaction)
if cache_key in ctx.deps.mcts_cache:
    ctx.deps.mcts_cache_hits += 1
    cached_result = ctx.deps.mcts_cache[cache_key]
    # Reuse MCTS tree or result
else:
    ctx.deps.mcts_cache_misses += 1
    # Run fresh MCTS
    result = mcts_engine.search(...)
    ctx.deps.mcts_cache[cache_key] = result
```

**Tasks:**
- [ ] Add `MCTSCacheEntry` model with tree state + metadata (REQ-SM-013)
- [ ] Implement cache key computation (merchant + amount bucket)
- [ ] Update classification tool to check cache before MCTS
- [ ] Update fraud detection tool to check cache
- [ ] Add cache size monitoring (invalidate if >50MB) (REQ-SM-014)
- [ ] Ensure cache cleared at session start (REQ-SM-015)
- [ ] Log cache hit rate to Logfire (REQ-SM-018)

### Phase 3: Logfire Session Tracking (OBSERVABILITY)

**Goal**: Add session_id to all spans and aggregate session metrics

**Tasks:**
- [ ] Add session_id to root span attributes (REQ-SM-016)
- [ ] Propagate session_id to all child spans (REQ-SM-017)
- [ ] Create `session_metrics` dict in SessionContext
- [ ] Aggregate: total_cost_usd, execution_time_seconds, cache_hit_rate, tool_failure_count (REQ-SM-018)
- [ ] Add `session_completed` Logfire span on Tool 4 completion (REQ-SM-019)
  - Attributes: final_status, output_csv_path, transactions_processed, peak_memory_mb

### Phase 4: Memory Management (CLEANUP)

**Goal**: Explicit memory cleanup after pipeline completion

**Tasks:**
- [ ] After Tool 4 completion, delete intermediate DataFrames (REQ-SM-010)
  ```python
  del ctx.deps.filtered_transactions
  del ctx.deps.raw_csv_data
  ctx.deps.mcts_cache.clear()
  gc.collect()
  ```
- [ ] Log memory freed to Logfire with `memory_freed_mb` metric
- [ ] Add `session_cleanup` span

### Phase 5: Immutability & Best Practices (CODE QUALITY)

**Goal**: Follow Pydantic best practices for context updates

**Current (❌ WRONG):**
```python
ctx.deps.results['filtered_df'] = new_df  # Mutates dict
```

**New (✅ CORRECT):**
```python
# Immutable update pattern
ctx.deps = ctx.deps.model_copy(update={
    'filtered_transactions': new_df,
    'execution_log': ctx.deps.execution_log + [log_entry]
})
```

**Tasks:**
- [ ] Audit all tool implementations for in-place mutations (REQ-SM-022)
- [ ] Replace with `model_copy(update=...)` pattern
- [ ] Add type hints to ensure SessionContext typing

### Phase 6: Testing (VALIDATION)

**Goal**: Unit tests for session memory behavior

**Test Cases:**
- [ ] Test Tool 2 can read Tool 1's output
- [ ] Test Tool 3 can read Tools 1 & 2 outputs
- [ ] Test MCTS cache hit/miss logic
- [ ] Test cache invalidation at session start
- [ ] Test session_id uniqueness across runs
- [ ] Test memory cleanup after Tool 4
- [ ] Test Logfire span hierarchy includes session_id
- [ ] Test immutable context updates don't affect prior state

---

## Deferred Requirements

### REQ-SM-011: Concurrent Session Safety

**Status**: Not applicable to current single-CSV batch processor

**Future Scenario**: If agent becomes an API service processing multiple CSVs concurrently:
```python
# Future API endpoint
@app.post("/analyze")
async def analyze_csv(file: UploadFile):
    session_context = SessionContext(
        session_id=str(uuid.uuid4()),
        csv_file_name=file.filename,
        raw_csv_data=pd.read_csv(file.file),
        config=load_config(),
        ...
    )
    result = await run_analysis(session_context)  # Each request gets own context
    return result
```

**Thread Safety Strategy**:
- Each API request creates new SessionContext instance
- No shared global state (already true in current design)
- DataFrames are immutable within session (REQ-SM-022 helps here)

**Decision**: Implement when/if API mode is needed. Current design is already thread-safe by accident (no globals).

### REQ-SM-012: Session Timeout

**Status**: Not applicable to batch processor

**Reason**:
- Sessions run start-to-finish in 2-10 minutes
- No idle time (tools run sequentially without pauses)
- No user interaction that could abandon session

**Alternative**: Add **maximum execution timeout** for hung sessions:
```python
# In run_analysis()
with timeout(max_seconds=600):  # 10 minute hard limit
    result = run_pipeline(ctx)
```

**Decision**: Optional safety feature, not a session memory requirement.

### REQ-SM-020: Session Debug Mode

**Status**: Nice-to-have, not critical for MVP

**Potential Implementation**:
```python
if os.getenv("SESSION_DEBUG") == "true":
    # Disable MCTS cache
    ctx.deps.mcts_cache_enabled = False

    # Verbose MCTS logging
    for iteration in mcts_iterations:
        telemetry.span(f"mcts_iteration_{iteration}", ...)

    # Dump context snapshots
    ctx.deps.save_snapshot(f".debug/session_{session_id}_after_tool_{tool_name}.json")
```

**Decision**: Implement after core session memory is working.

---

## Success Metrics

### Before Session Memory Implementation

- **Cost per CSV**: ~$2.50 (no MCTS cache reuse)
- **Processing time**: ~5 minutes for 100 transactions
- **Memory usage**: Unknown (no tracking)
- **Debuggability**: Hard to trace tool outputs without session_id

### After Session Memory Implementation

- **Cost per CSV**: ~$1.50 (40% reduction via MCTS cache)
- **Processing time**: ~3 minutes (40% reduction via cache)
- **Memory usage**: Tracked and cleaned up explicitly
- **Debuggability**: Easy span filtering by session_id in Logfire
- **Cache hit rate**: 60-70% for typical CSVs with repeated merchant categories

---

## Next Steps

1. **Create SessionContext model** in `src/session_context.py`
2. **Implement MCTS caching logic** in classification and fraud detection tools
3. **Add session_id to Logfire spans** in telemetry layer
4. **Update run_analysis()** to use SessionContext instead of AgentDependencies
5. **Write unit tests** in `tests/test_session_memory.py`
6. **Benchmark performance** before/after to validate cost savings

---

## Appendix: Code Changes Summary

### Files to Create
- `src/session_context.py` - SessionContext Pydantic model + utilities

### Files to Modify
- `src/agent.py` - Replace AgentDependencies with SessionContext
- `src/mcts_engine.py` or `src/mcts_engine_v2.py` - Add caching hooks
- `src/telemetry.py` - Add session_id propagation
- `src/csv_processor.py` - Add memory cleanup on save

### Files to Add Tests
- `tests/test_session_memory.py` - Session isolation and memory passing tests
- `tests/test_mcts_cache.py` - Cache hit/miss and invalidation tests
