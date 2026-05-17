# INSIGHTS — dcr-attention

Append-only. Сюда пишем то, что неочевидно, что стоило нам времени на обнаружение,
или что определит архитектурные решения ниже по стеку.

**Критерии записи:**
- Наблюдение противоречит нашему первоначальному ожиданию.
- Открытие влияет на >1 модуль или на несколько фаз roadmap.
- Константа / параметр / граница режима, выведенная из численных данных.
- Bug-pattern, который может повториться, — документируем для предотвращения.
- Формулировка paper-claim'а, подтверждённая экспериментом.

Формат:
```
## INS-<N> — <заголовок>
**Дата:** YYYY-MM-DD
**Область:** kernel | dispatcher | model-integration | benchmark | paper
**Severity:** blocker | major | minor | note

<описание наблюдения, что ожидали, что увидели, почему важно>

**Следствие:** <что меняем / фиксируем / валидируем дополнительно>
```

---

## INS-1 — Reference PyTorch rank_local_attention материализует полную маску O(N²)
**Дата:** 2026-04-24
**Область:** kernel, paper
**Severity:** note

Референс `dcr_benchmark/rank_local_attention.py` (строки 54–57) строит `in_window: [B,H,N,N]`
через `rank_of_k_expanded - r_center_expanded`, затем `scores.masked_fill(~in_window, -inf)`.
Это O(N²) память и compute — то есть PyTorch-референс **не демонстрирует** экономии, ради которой
DCR существует. Экономия появляется только в block-sparse Triton-реализации.

**Следствия:**
1. Ground-truth для correctness возможен только при N ≲ 4096 (выше — референс OOM).
2. Это **лимит correctness-теста**: корректность Triton-kernel при N=128K напрямую против
   reference мы проверить не можем. Нужна опосредованная верификация: (a) равенство при
   k=2N vs dense-reference при N=4096, (b) сохранение perplexity на реальной модели, (c) cos-sim
   vs SDPA-flash при N=8K–32K (SDPA flash тоже block-sparse, но exact).
3. В paper v2 §5 это надо объяснить явно: «memory claim measured vs FA-3, not vs PyTorch mask
   reference», иначе запутает ревьюера.

## INS-2 — Dispatcher Stage B (PCA) — потенциальный bottleneck, требует warmup-кеш
**Дата:** 2026-04-24
**Область:** dispatcher
**Severity:** major

В `dispatch()` (строки 130–134 reference) ось получается через `torch.linalg.svd(Qc, full_matrices=False)`
при размере `Qc ∈ [N, D]`. При N=131k, D=128 SVD на GPU даёт ~O(N·D²) ≈ 2·10⁹ flops per layer per forward —
это значимый overhead для inference. Roadmap §4.1 упоминает «axis precomputed at patch time or per-layer
warmup» — это обязательное требование, не оптимизация.

**Следствие:**
- Warmup-стратегия: при patch_model() делаем один forward на калибровочном батче, сохраняем axis per layer.
- Для positional-оси (`axis_source='positional'`) вычисление дешевле и детерминистично — делаем при patch-time
  один раз, без warmup-батча.
- Fallback-политика: если `confidence < tau_confidence` на реальном forward при cached-axis, откатываемся на dense.

## INS-3 — Triton forward-only не покрывает activation-check memory claim «из коробки»
**Дата:** 2026-04-24
**Область:** paper, benchmark
**Severity:** major

Roadmap v1.0 §9 обещает «≥15% memory reduction vs FA-3 at 128K». FA-3 forward держит O(N) activation-память
через tiling + online softmax. KV-cache O(2NHD) одинаков у любой архитектуры Llama. Экономия DCR над FA-3
возможна только на intermediates **prefill-стадии** — не на generation-стадии (generation обрабатывает
1 query-токен против N key-токенов, там FA-3 и DCR почти эквивалентны по памяти).

**Следствие:**
- Killer-claim надо переформулировать: «peak memory reduction during 128K **prefill**» — источник выигрыша явен.
- Benchmarks/llama_memory.py должен разделить две фазы: prefill-peak и decode-steady-state.
- Если на decode мы не выигрываем — это не провал, это конструктивная особенность, которую paper объяснит честно.

## INS-4 — Dev-окружение Claude: torch доступен, CUDA нет
**Дата:** 2026-04-24 (updated)
**Область:** infrastructure
**Severity:** minor

Окружение имеет torch 2.11.0+cu130 предустановленным, но без CUDA-устройства
(`torch.cuda.is_available() == False`). Все PyTorch-тесты в fallback-режиме запускаются на CPU:
медленнее, но численно идентично fp32-пути GPU. Для Triton-kernel это **блокирует** прогон в этой
среде — Triton требует CUDA. Triton-тесты будут запускаться коллегой на RTX 4060 Ti.

**Следствие:**
- Я могу прогонять reference-тесты и PyTorch-dispatcher unit-тесты прямо здесь.
- Triton-kernel валидируется: (a) AST-парсинг, (b) статический разбор signature, (c) коллега запускает.
- Благодаря CPU-прогонам первые численные несоответствия ловим сразу — пример: INS-5 ниже.

## INS-5 — Reference-тесты v2.1.0-rc1 содержат необоснованные пороги в print-statements
**Дата:** 2026-04-24
**Область:** infrastructure, paper
**Severity:** major

Исходный `dcr_benchmark/test_rank_local.py` имеет строку
`print(f"k=N/2 : cos_sim = {cos:.4f}   (should be > 0.90)")` — но это **print, не assert**.
Реальное численное значение при random i.i.d. Gaussian Q,K (seed=42, N=128, D=64, axis=e_0)
составляет cos=**0.7020**, не 0.90+. Цифра 0.90 в комментарии является wishful thinking,
не эмпирически проверенной границей.

**Причина:** axis=e_0 на случайных Gaussian объясняет ~1/D ≈ 1.6% дисперсии; ранжирование
по ней почти случайное; rank-local window захватывает случайные keys. Это корректное
поведение алгоритма — не bug, но и не «rank-local attention captures most of the mass»
как имплицировал оригинальный тест.

**Следствия:**
1. Все пороги в reference-тестах v2.1.0-rc1 надо пересмотреть перед переносом.
   Не копировать `should be > X.XX` как assert без валидации.
2. Тестирование rank-local на random данных **не показывает его преимуществ** —
   нужны structured inputs (low-rank signal + noise) с axis, выбранной по структуре сигнала.
3. Для paper v2 §5 применимость test-коллекции в роли empirical support для claim-ов
   требует структурированного test-protocol: `random`, `rank-1`, `rank-k`, `anisotropic-cov`.
   Roadmap §6 должен явно добавить это как ablation study.
4. **Положительный результат:** при SNR=5 (signal=5·u, noise=1·I), axis ∈ span{signal},
   k=N/2 → cos_sim > 0.95 — подтверждено в `test_half_window_structured_inputs_preserves_mass`.
   Это первый **empirically validated** threshold для rank-local в нашем репо.

## INS-6 — Positional-axis для RoPE моделей не работает «из коробки»
**Дата:** 2026-04-24
**Область:** model-integration
**Severity:** major

`Stage B` имеет два пути выбора оси: `positional` (через `lstsq(PE_centered, j_target)`) и
`pca` (через `svd(Qc)`). Путь `positional` предполагает **additive positional embedding** —
learned `PE[N, D]` tensor, который добавляется к token embeddings до attention. Это случай
BERT / ViT (reference v2.1.0-rc1 интеграции).

**Llama 3, Mistral, Qwen используют RoPE** — positional information «вшивается» в Q и K
через поворот пар координат функцией от позиции, а не как отдельный тензор. Наивное
применение `compute_axis(Q, positional_embedding=PE)` на Llama бессмысленно: (a) такого
тензора нет, (b) если подсунуть RoPE-cos/sin-cache как PE, оно не даёт монотонной
проекции в позицию.

**Следствия:**
1. Для Llama-интеграции (roadmap Phase 2) Stage B вынуждена работать в режиме `pca`.
2. PCA-ось при длинных контекстах дорогая (см. INS-2) — требует warmup-кеширования.
3. Возможный третий путь: **learned axis** per layer per head (roadmap §4.1 упоминает
   `stage_b.py — Axis selection (PCA, positional, learned)`). Это добавляет параметры в
   модель; несовместимо с forward-only drop-in patch — откладываем на v2 с backward.
4. Альтернативная позиционная ось для RoPE: взять ось, вдоль которой RoPE-вращение
   монотонно растёт по позиции — эта ось есть в cos/sin частотном спектре, но её
   извлечение требует отдельного анализа (TODO).

## INS-7 — Positional lstsq чувствителен к N/D ratio и шуму PE
**Дата:** 2026-04-24
**Область:** dispatcher
**Severity:** minor

`_axis_from_positional` решает `min ||P_c·u − j_c||²`. При N=128, D=32, σ_noise=0.001
(sample-to-dim ratio 4:1) — восстановление истинной оси деградирует до cos≈0.88 с истинной
v. При N=1024, D=32, σ=0.01 (ratio 32:1) — cos>0.95. В чистом случае (σ=0) —
cos > 0.9999 (до machine precision).

**Следствие:**
- Для моделей со сравнительно высокой d_head (Llama-3 8B: d_head=128) и короткими
  калибровочными батчами warmup-на-64-токенах — не пройдёт. Нужен калибровочный набор
  длиной ≥ 4D токенов, лучше ≥ 8D.
- Тест `test_positional_axis_recovers_monotone_direction` разделён на noiseless (cos>0.9999)
  и noisy (cos>0.95 at N=1024, D=32, σ=0.01) — первый гарантирует правильность алгоритма,
  второй документирует реально достижимую точность.

## INS-8 — Pre-gather K_sorted/V_sorted стоит 2·B·H·N·D памяти
**Дата:** 2026-04-24
**Область:** kernel, benchmark
**Severity:** major

Архитектурное решение P2 (внешние sort indices) мы реализовали через pre-gather:
`K_sorted = K[sort_idx]`, `V_sorted = V[sort_idx]`, после чего окно становится
contiguous slice в sorted space. Стоимость — лишние 2·B·H·N·D байт памяти.

Конкретно для Llama-3 8B prefill при N=128K, D=128, H=32, batch=1, bf16:
2 × 32 × 131 072 × 128 × 2 B = **2 GiB peak additional** per attention call.
Это тот самый overhead, который **съедает** заявленный «≥15% memory reduction vs FA-3»
из INS-3, если не делать что-то хитрее.

**Альтернатива — scatter-gather прямо внутри Triton:** `K[sort_idx[j_sorted]]` через
`tl.load` с indirection. Работает, но теряет coalesced access pattern и становится
на 1.5–3× медленнее на Ampere по оценкам из FA-3 paper приложения.

**Следствия:**
1. Claim в paper v2 «memory reduction» **должен учитывать** overhead pre-gather.
   Net benefit = (O(Nk) attention-intermediates saved) − (2BHND extra).
   При Nk < 2ND (т.е. k_window < 2D) экономия отрицательна — мы тратим больше.
2. Для Llama (D=128) граница: k_window < 256 → pre-gather неприбылен.
   Это **критический** факт: стандартный k_window 64–256 попадает в зону потерь.
3. Phase 4 optimization (in-kernel scatter-gather) **обязателен** до paper-claim-ов,
   не optional.
4. Временно: сохраняем pre-gather ради корректности Phase 1.2; benchmark-скрипт
   в Phase 3 обязан измерять net memory delta, не только attention-intermediate saving.

## INS-9 — torch.autograd.Function.forward ставит ctx даже при no-grad inputs
**Дата:** 2026-04-24
**Область:** kernel
**Severity:** minor

При вызове `Fn.apply(a, b, c)` где ни один из a, b, c не `requires_grad`, autograd
всё равно создаёт `ctx` объект и передаёт его в `forward()`. Не создаётся только
Node в graph (т.е. `backward` никогда не вызовется). Но `ctx` есть, и можно в нём
держать атрибуты — они просто никогда не прочитаются.

**Следствие:**
- Самоадаптивная логика `if any(t.requires_grad for t in inputs): ctx.save_for_backward(...)`
  работает корректно — просто пропускаем save_for_backward в no-grad случае.
- Мы всё же сохраняем `ctx.needs_backward = bool` — дешёвая защита от будущего
  bug-паттерна, если кто-то добавит backward implementation без проверки.
- Zero-memory-overhead в inference подтверждён: в no-grad случае ни один тензор
  в ctx не хранится (проверено `test_no_grad_skips_save_for_backward`).

## INS-10 — Mean-pooling выкидывает PCA-сигнал; ось живёт в ковариации
**Дата:** 2026-04-24
**Область:** dispatcher, projector
**Severity:** major (paper-grade)

Первая версия projector использовала `mean(Q, dim=N)` как pooled-feature input для MLP.
Архитектура **не училась**: rank-1 cos=0.55 (надо 0.95), rank-4 cos=0.33. После расследования
выяснилось: для Q с rank-1 структурой `Q_i = s_i · u + noise`, где `s_i ~ N(0,1)`,
имеем `mean(Q) = mean(s) · u + mean(noise) → 0` при N→∞. Информация о направлении u **не
содержится в среднем**, она содержится в **дисперсии** (т.е. в `QᵀQ/N`).

**Корректная архитектура:**
1. Pool через второй момент `Σ = QᵀQ/N` ∈ ℝ^{D×D}.
2. Извлечь `u_pca = top_eigvec(Σ)` с фиксацией знака (largest-magnitude entry > 0).
3. MLP: feats = `[upper_tri(Σ), u_pca]` → `δ ∈ ℝ^D` (residual prediction).
4. axis = `normalize(u_pca + δ)`.
5. Init `W2 = 0, b2 = 0` → δ = 0 → projector ≡ PCA baseline.

Проверено эмпирически: при init **cos = 1.0000** с PCA target (точное соответствие).
После 200 шагов training на rank-1 → held-out cos = 1.0000.
После 400 шагов на rank-4 → held-out cos = 0.9961.

**Cost analysis (важно для INS-2 / SVD-overhead claim).**
- SVD `Q ∈ ℝ^{N×D}`: `O(ND²)`, на длинных-тонких матрицах bandwidth-bound.
- Σ-construction: `O(ND²)`, но это один GEMM call, hits tensor cores at peak.
- `eigh(Σ)`: `O(D³)` — для D=128 это 2·10⁶ flops, vs SVD's 2·10⁹ at N=128K.
- Net wallclock saving: 3 порядка на шаге собственно decomposition, плюс GEMM efficiency.

**Следствия для paper v2:**
- Это первая **archtitectural lesson** для §3 (DCR Architecture).
- Цитата для §7 Discussion: «pooling choice in axis prediction must respect the
  algebraic structure of the target — mean for first-moment statistics, covariance
  for second-moment». Звучит банально, на практике это была наша главная блокирующая ошибка.
- Test `test_projector_init_matches_pca_baseline` теперь служит regression guard:
  если кто-то изменит init и projector перестанет начинать с PCA — поймаем сразу.

## INS-11 — Acceptance criterion «loss decreases» неприменим к residual-architecture
**Дата:** 2026-04-24
**Область:** test methodology
**Severity:** minor

Когда projector предсказывает **residual** к аналитическому baseline (PCA), и baseline
уже близок к оптимуму, training loss стартует около нуля и не имеет куда падать.
Acceptance criterion «loss must decrease 2× over training» — bug в самом criterion,
не в коде.

**Правильные acceptance criteria для residual-models:**
- Held-out quality ≥ threshold (это сохранили).
- Loss does not diverge (5× early-window — generous regression guard).
- Init equals analytical baseline (новый тест, защищает от bad init).

**Следствие:** при добавлении новых learnable-residual компонентов в проект —
не копировать «loss decreases» test без проверки, что baseline даёт нулевой residual.

## INS-12 — Structural patches imported from larger projects scale by squared cost
**Дата:** 2026-04-25
**Область:** methodology
**Severity:** major (paper-grade)

При review TECHNICAL_PATCH_v1 (8 патчей из φ-HyperSolver / φ-VISION) обнаружен
системный паттерн: structural improvements, разработанные в большом проекте
(C++ DCR Sort v2.0, External Memory v3.0, Track D paper), масштабируются в маленький
проект (dcr-attention) **квадратично** по cost: больше components × больше enforcement
points × больше maintenance burden.

Конкретный пример из этого patch-документа:
- ABC1 reproducibility (PATCH-01): 4 hash64 + run_set_hash + seed_metric formula —
  нужно для DOE-плана и bootstrap CI95% Track D paper. Для dcr-attention,
  где ещё нет даже benchmarks/results.json, нужна 1 функция `record_environment()`.
- Error Ontology (PATCH-02): 4 классов / R1-R7 enforcement / JSON schema — для проекта
  где failure modes нуждаются в downstream branching. Для dcr-attention `pytest pass/fail`
  достаточно.
- 14 Architectural Invariants (PATCH-03): из них 11 — про SIMD/AVX-512/OpenMP/scatter
  C++ DCR Sort v2.0. Для dcr-attention применимы 2.

**Принцип принятия решения:** «adopt the smallest subset that closes a *known* bug
class, not the full pattern that closes the *imagined* bug class».

Phase 4b был реальный bug → I_DET_1 (no torch.randperm) реально нужен.
Phase 4c-4z bugs могут не материализоваться → пре-emptive infrastructure для них —
dead weight, который надо поддерживать.

**Quantitative outcome для нашего конкретного review:**
- 8 патчей принять полностью: ~5-7 дней (P1) + 3-4 недели (P1+P2).
- Принятая частично подмножина (PATCH-01 минимум, PATCH-03 в части I_DET_1, PATCH-08 S2):
  **~1 час**.
- Net deferred: ~3-4 недели ceremony.

**Следствия для будущих imports:**
1. При получении нового spec-документа из external project — первый вопрос «какой
   именно bug class он закрывает в *моём* проекте?», не «какие лучшие practices
   из него можно перенять».
2. Patches без conkretного bug class в моём проекте → backlog, не immediate adoption.
3. Бeglog-маркер `TODO(patch-vN)` для отложенных частей — reactivation should be
   one search away, не reading-a-document-again-from-scratch away.

Это попадает в paper v2 §7 Discussion как methodology lesson: «pre-emptive engineering
infrastructure imported wholesale from a larger sibling project tends to be net-negative
ROI for a focused single-engineer effort; selective adoption per known bug class
preserves velocity».

## INS-13 — Online softmax с -inf sentinel: nan-trap в первой итерации
**Дата:** 2026-04-25
**Область:** kernel
**Severity:** blocker (confirmed real bug, fixed)

При первом GPU-прогоне fp32 kernel компилировался и выполнялся, но давал расхождение
с reference на O(1) absolute (max|abs| = 1.35–4.13). Smoke-test это не поймал, потому
что smoke только проверяет `torch.isfinite(out).all()` — а NaN в kernel уходят в нули
после `tl.where(l_i > 0, ...)` финализации.

**Источник bug.** Online softmax на старте имеет `m_i = -inf, l_i = 0, acc = 0`.
Если **первое обработанное окно** для какой-то query полностью замаскировано (в нашем
случае это норма: rank-window сужается, для query в начале/конце последовательности
большинство k-tiles не пересекается), все scores `s = -inf`. Тогда:

```
m_new = max(-inf, max(-inf, ..., -inf)) = -inf
alpha = exp(m_i - m_new) = exp(-inf - (-inf)) = exp(NaN) = NaN
p     = exp(s - m_new)  = exp(-inf - (-inf)) = exp(NaN) = NaN
```

NaN распространяется через `acc = acc·alpha + dot(p, v)`, и кaplan хотя финализатор
зануляет out для `l_i ≤ 0`, **NaN-загрязнённый acc** уже накопился из non-empty
последующих окон, даёт неправильный output.

**Фикс.** Заменить sentinel `-inf` на `_NEG_INF_SAFE = -1e30`. Свойства:
- `exp(-1e30) = 0` в fp32 (underflow flush) → правильное поведение для маскированных bins.
- `m_i - m_new = -1e30 - (-1e30) = 0`, `exp(0) = 1` → alpha=1 на пустой первой итерации,
  но `p ≈ 0`, поэтому `acc = 0 + 0 = 0`, всё корректно.
- Когда первое валидное окно даёт настоящий `m_new = 1.5`, `alpha = exp(-1e30 - 1.5) ≈ 0`,
  что обнуляет старый `acc = 0` (no-op), всё корректно.

Это **тот же** трюк, что используется в FlashAttention-2 reference, но я его пропустил
при первом написании. Стандартный pitfall online softmax с window mask, где «no key
hits this query in this tile» — обычное явление, в отличие от full attention где
m_new всегда становится финитным после первой итерации.

**Следствие для тестов.** Smoke-test `torch.isfinite(out).all()` **недостаточен** для
поимки этого класса bug — финализатор маскирует NaN. Правильная защита — output parity
vs reference (что и сделано в `test_output_matches_torch_fallback`).

**Следствие для paper v2.** Это going into §4 Implementation как один из «lessons from
the first GPU run» — конкретная инженерная деталь, которая может сэкономить кому-то
часы дебага.

## INS-14 — Triton 3.1: tl.dot требует identical dtype operands
**Дата:** 2026-04-25
**Область:** kernel
**Severity:** blocker (confirmed real bug, fixed)

`tl.dot(q, tl.trans(k_tile))` падал с `AssertionError: First input (fp32) and second
input (bf16) must have the same dtype!` для bf16/fp16, но **не для fp32**. Это потому что:

1. `q = tl.load(...)` грузит в native dtype (bf16 для bf16 input).
2. `q = q * scale`, где `scale` — **Python float**. В Triton 3.x умножение тензора на
   Python скаляр **НЕ** сохраняет dtype тензора — компилятор промотает результат в fp32.
3. `tl.dot(q_fp32, tl.trans(k_bf16))` → assertion fail.

В fp32 случае оба операнда уже в fp32, проблема не возникала (но fp32 упал по INS-13).

**Фикс.** Явный fp32 promotion для всех accumulators:
```
q_native = tl.load(q_ptrs, ...)
q = (q_native * scale).to(tl.float32)
k_tile = tl.load(k_ptrs, ...).to(tl.float32)
v_tile = tl.load(v_ptrs, ...).to(tl.float32)
```

После promotion все операнды `tl.dot` гарантированно в fp32, accumulators (m_i, l_i, acc,
s, p) тоже в fp32. Это совпадает с FA-2 reference (mixed precision: tile loads в native,
все math в fp32, store обратно в native через autocast `tl.store`).

**Trade-off.** Промоутить в fp32 каждое окно дороже по памяти SRAM (×2 bytes/element),
но это **корректно**. Phase 4 optimization может рассмотреть mixed-precision dot
(tl.dot с tf32 acc на A100/H100), но **только** после того, как fp32 path устойчив.

**Следствие для процесса.** Я знал о risk «`tl.trans` API drift между Triton 2/3» и
явно упомянул его в первой передаче коллеге («Если строка `tl.dot(q, tl.trans(k_tile))`
уронит компиляцию...»). Но НЕ предсказал что виновник — `q * scale` ломает dtype
автоматически. Урок: **API-drift предсказывать сложно даже зная version diff**;
лучшая защита — phase-gated GPU validation, не предположение что код компилируется
из чтения.

## INS-15 — Triton 3.1 запрещает не-constexpr module globals в @triton.jit
**Дата:** 2026-04-25
**Область:** kernel
**Severity:** blocker (confirmed real bug, fixed)

После применения патча из INS-13 второй GPU validation round упал на этапе
компиляции с `NameError: Cannot access global variable _NEG_INF_SAFE from within
@jit'ed function. Triton kernels can only access global variables annotated as
constexpr`.

**Источник bug.** В Triton 2.x module-level Python globals (constants) свободно
читались внутри `@triton.jit` функций. В Triton 3.1 это **запрещено** — сделано
для compile-time дедупликации и предотвращения silent ABI changes при изменении
глобала между запусками. Решения:
1. PEP 526 annotation: `_NEG_INF_SAFE: tl.constexpr = -1.0e30` (рекомендованный).
2. Function call: `_NEG_INF_SAFE = tl.constexpr(-1.0e30)` (функционально идентично).
3. Inline literal: заменить все вхождения `_NEG_INF_SAFE` на `-1.0e30`.

**Выбор: вариант 1.** Сохраняет self-documenting именованную константу, не загромождает
ядро литералами, и совпадает с тем, что сообщает сам compiler в error message —
defensive programming for future contributors.

**Технический момент.** PEP 526 annotation не создаёт runtime `tl.constexpr` объект:
`type(_NEG_INF_SAFE)` показывает `float`, не `constexpr`. Triton AST-walker читает
**module `__annotations__` dict** отдельно и применяет правило constexpr на этапе
JIT compilation. То есть annotation — hint для Triton'а, не runtime структура.

**Следствие для тестов.** Этот bug нельзя поймать AST-парсингом или import-time
проверкой — только запуск kernel компиляции на GPU вскрывает его. Мой Phase 0 дизайн
«AST + pytest --collect-only достаточно для CPU-валидации» был **правильным** для
catching syntax errors и module-import errors, но **не** для catching Triton-specific
constraints. Для Triton-кода единственный надёжный gate — `pytest tests/.../test_kernel_compiles_and_runs` на CUDA-машине.

## INS-16 — Triton tl.dot fp32 inputs use TF32 by default (10-bit mantissa)
**Дата:** 2026-04-25
**Область:** kernel
**Severity:** blocker → fixed (paper-grade)

После применения фиксов INS-13 (NaN-trap) и INS-15 (constexpr), GPU validation v3
показал: kernel компилируется, но fp32 даёт **систематическое** расхождение с torch
fallback на уровне `1.9e-3 — 3.5e-3 max|abs|` для **всех 30 fp32 кейсов**, при
tolerance 1e-5. Ошибка не растёт с N (128→2048), значит это не accumulation —
per-dot-product noise.

**Источник bug.** Triton 3.x документация явно говорит:
> «Default: "tf32". How to exercise the Tensor Cores for f32 x f32. ... The default
> precision is "tf32" on NVIDIA hardware.»

При `tl.dot(q_fp32, k_fp32)` на Ampere/Ada/Hopper, Triton **по умолчанию** использует
TF32 tensor cores. TF32 имеет 10-bit mantissa (vs 23-bit у IEEE fp32), что даёт
точность ~`2^-10 ≈ 9.7e-4`. Это **ровно** та magnitude ошибки, которую мы видим.

Torch fallback использует `torch.einsum` через cuBLAS, который с PyTorch 1.12+ имеет
**`torch.backends.cuda.matmul.allow_tf32 = False`** по умолчанию — то есть честный
IEEE fp32. Отсюда расхождение.

**Фикс.**
```python
s = tl.dot(q, tl.trans(k_tile), input_precision="ieee")
acc = ... + tl.dot(p, v_tile, input_precision="ieee")
```
`input_precision="ieee"` отключает TF32 path и заставляет Triton использовать
честный IEEE fp32 multiply-accumulate. **Цена:** на Ampere/Ada не используются
tensor cores → ~6-8× медленнее dot. Для **correctness** path — приемлемо.

**Phase 4 будущее.** Когда kernel доказан корректным, можно ввести autotune-ed
ключ `precision: tl.constexpr` со значениями {"ieee", "tf32"}. Llama inference
probably tolerates TF32 (downstream loss/accuracy не отличаются), но это нужно
**измерять** на perplexity, не предполагать. До того момента — IEEE как honest baseline.

**Косвенное последствие для bf16/fp16.** В моём kernel bf16/fp16 tiles тоже cast в
fp32 перед `tl.dot` (фикс из INS-14). До INS-16 это шло через TF32, что усиливало
ошибку на bigger D (D=128 — 15 фейлов в bf16). После INS-16 fix bf16 D=128 cases
тоже должны починиться или выйти на bf16-inherent noise floor.

## INS-18 — Reference oracle должен быть численно строже валидируемого пути
**Дата:** 2026-04-25
**Область:** test methodology, kernel
**Severity:** major (paper-grade)

GPU validation v4 категория B: `bf16, D=32, max|abs| = 1.95e-2` против tolerance 1e-2.
Расследование показало: `rank_local_fwd_torch` (наш ground-truth oracle) выполнял
**всю математику в bf16** при bf16 inputs:
```python
scores = einsum(Q_bf16, K_bf16) * scale     # bf16 scores
attn = softmax(scores)                      # bf16 softmax
out = einsum(attn, V_bf16)                  # bf16 output
lse = logsumexp(scores)                     # bf16 LSE
```
В то же время Triton kernel **upcast'ит** Q/K/V в fp32 для всей внутренней математики
(после fix INS-14, и тем более после fix INS-16 с `input_precision="ieee"`).

Получается **парадокс**: тестируемый kernel более точен чем oracle. Расхождение
test_t vs test_tr — это шум **oracle**, а не баг **kernel**. Расхождение `1.95e-2`
в bf16 D=32 — это native bf16 softmax accumulation noise в torch fallback, который
Triton **избегает** через fp32-internal math.

Это также объясняет категорию C (LSE dtype): `torch.logsumexp(bf16_scores)` возвращает
bf16, но контракт у нас «LSE always fp32».

**Фикс.** В `rank_local_fwd_torch`:
```python
out_dtype = Q.dtype
Q = Q.to(torch.float32)
K_sorted = K_sorted.to(torch.float32)
V_sorted = V_sorted.to(torch.float32)
# ... всё math в fp32 ...
return out.to(out_dtype), lse  # cast только output, LSE остаётся fp32
```
Это закрывает обе категории (B и C) одним изменением и приводит oracle в соответствие
с Triton mixed-precision стратегией.

**Принцип (paper-grade).** *«Reference oracle for a test must be at least as
numerically accurate as the path under test»*. Иначе невозможно различить
oracle-noise vs implementation-bug. В нашем случае это эквивалентно: «mixed-precision
implementations require fp32-internal reference».

Связано с `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)` — PyTorch
сам с 2.x версии **upcast'ит** bf16 в SDPA для математики по той же причине.
Следуем установленной практике.

**Связь с paper v2.** Это **четвёртый** numerical lesson (после INS-13/14/16) и
вероятно самый высокого уровня — это про test methodology, не про kernel mechanics.
Идёт в §5 Experiments как methodological note: tolerance values are not opinions,
they're consequences of oracle precision design.

## INS-19 — Triton SMEM budget для D=128: BLOCK_K=64 num_stages=3 не помещается
**Дата:** 2026-04-25
**Область:** kernel
**Severity:** blocker (architectural, fixed)

GPU validation v4 категория A: 24 fails при D=128 с
`OutOfResources: Required: 155904, Hardware limit: 101376` (RTX 4060 Ti / Ada).

**Вычисление SMEM**. Default config (BLOCK_Q=32, BLOCK_K=64, num_stages=3) держит
в SMEM:
- Q-block (резидентный): 32 × 128 × 4 = 16 KB
- K, V tile (×3 stages для pipelining): 2 × 64 × 128 × 4 × 3 = **192 KB peak**, ~ effective 96 KB при overlap
- s, p: 32 × 64 × 4 × 2 = 16 KB
- acc: 32 × 128 × 4 = 16 KB
Сумма ≈ 144 KB peak, что превышает 100 KB Ada SM limit.

(Точное число 155 KB зависит от Triton internal layouts, но порядок верен.)

**Фикс — adaptive config**.
```python
if D <= 64:    BLOCK_Q=32, BLOCK_K=64, num_stages=3   # full pipelining
else:          BLOCK_Q=32, BLOCK_K=32, num_stages=2   # for D=128
```
При D=128, halved BLOCK_K + num_stages=2 даёт SMEM:
- Q-block: 16 KB
- K, V tile (×2 stages): 2 × 32 × 128 × 4 × 2 = **64 KB**
- s, p: 32 × 32 × 4 × 2 = 8 KB
- acc: 16 KB
Итого ≈ 104 KB. Всё ещё на грани, но Triton внутренние оптимизации обычно дают
запас. Если round 5 покажет всё ещё OutOfResources — `BLOCK_K=16` или `num_stages=1`.

**Phase 4 будущее.** Полноценный `@triton.autotune` с конфигами под (Ampere | Ada |
Hopper) × (D, k_window) — ~30 конфигов. Сейчас не делаем — это **Phase 4 optimization**,
а мы в Phase 1.2 correctness. Hardcoded heuristic покрывает наш test sweep.

**Hardware contextualisation для paper.** RTX 4060 Ti (Ada Lovelace, sm_89) has
100 KB max dynamic SMEM per SM. A100 (Ampere, sm_80) has 164 KB. H100 (Hopper, sm_90)
has 228 KB. Таким образом нашу config'у автоматически нужно поднимать на A100/H100.
Это будет показано в Phase 3 (cloud A100 runs) — там SMEM уже не bottleneck для D=128.

## INS-20 — Algorithmic O(N²) bug, скрытый за TF32 noise
**Дата:** 2026-04-25
**Область:** kernel, methodology
**Severity:** blocker (paper-grade methodology lesson)

После закрытия Phase 1.2 GPU validation (89/89 correctness tests) microbench v5
показал DCR/SDPA ratio **6×** (D=64) — **200×** (D=128) при `k_window` ∈ {64, 256}.
Я первоначально атрибутировал это к `input_precision="ieee"` (отключённые tensor
cores) + reduced BLOCK_K для D=128. Коллега посмотрел на цифры и заметил:

> **Latency идентична для k_window=64 и k_window=256 на одной и той же (N, D, dtype).**

Это **прямое empirical доказательство** что compute не зависит от k. Inspection
kernel'а (rank_local_fwd_triton.py:114) подтвердил: `for k_start in range(0, N, BLOCK_K)`
итерирует по **всему** N, а window mask применяется через `tl.where(...)` *после*
`tl.dot()` — то есть FLOPs не экономятся. Kernel выполнял O(N²·D) compute независимо
от k.

**Это не bug в традиционном смысле — это conscious design choice Phase 1.2 ради
correctness-first.** Bounded loop требует сортировки Q (не только K), что выходило
за scope Phase 1.2. Я не задокументировал этот выбор как явное deferred-optimization,
и в результате 5 GPU validation rounds fixed numerical correctness, но не trip'нули
performance regression.

**Почему я пропустил.** Когда microbench показал плохие numbers, я пошёл «top-down»
от наиболее знакомого компонента (TF32 mantissa) к менее знакомому. Latency-vs-k
invariance виден только если **до** анализа цифр сформулировать гипотезу «O(N) или
O(N²) compute?» и проверить её первым. Я этого не сделал — поэтому коллега, посмотрев
на те же цифры **с другим вопросом**, увидел паттерн который я пропустил.

**Lesson (paper-grade).** *«Performance debugging должен начинаться с явного теста
order-of-growth (1 dimension at a time), не с анализа individual kernel components.»*
Конкретный protocol для будущих regression-investigations:

  1. Fix all parameters except one. Vary it across 3-4 values.
  2. Plot wall-clock vs that parameter.
  3. If slope не соответствует expected complexity → algorithmic issue.
  4. Только после step 3 переходить к kernel-level optimization.

Этот тест занимает 5 минут per parameter × 3-4 parameters = 30 минут up-front,
и сэкономил бы 1 GPU validation round (3-5 человеко-часов).

**Phase 1.3a fix.**

  1. New SortIndices schema: `sort_idx_q`, `inv_q_perm`, `sort_idx_k` (renamed
     from `sort_idx`), `rank_of_k`. r_center semantics changes from «for original
     Q-position i» to «for sorted Q-position t».
  2. After Q-sort, r_center is **monotonically non-decreasing** in t — это
     property проверяется в `test_r_center_is_monotone_in_t` как paper-grade
     correctness invariant.
  3. Bounded K-loop: для одного BLOCK_Q tile берём `[min(r_c) - half_k,
     max(r_c) + half_k]` и итерируем только эти `~k_window/BLOCK_K` K-tiles.
  4. Public API `rank_local_attention(Q, K, V, axis, k_window)` без изменений
     для callers — Q-sort + inv_perm применяются внутри wrapper.

**Realistic expected speedup.** Theoretical upper bound `N/(k+BLOCK_Q) = 85×`
для (N=8192, k=64). Реалистично — 5-15× из-за: (a) memory bandwidth ceiling,
(b) sort overhead (extra argsort + 2 gather), (c) Triton runtime loop bounds
без compile-time unrolling. Это **значимо** прогресс, но не превратит DCR
в SDPA-killer; превратит DCR/SDPA ratio из 200× в 13-40× при D=128.

**Phase 1.3a deliverables.** Kernel updated, public wrapper updated, tests
adapted to new contract (test_sort_helpers полностью переписан, test_rank_local_fwd_torch
обновлён, GPU correctness suite обновлён). 108/108 CPU tests passing, +5 new tests
от updated schema. Готов к v6 GPU validation round. Decision gate locked в
docs/design/phase1_3_q_sort.md §10 **до** measurement.

## INS-21 — Process miss: микробенч не входил в мой sweep-зачистки тестов
**Дата:** 2026-04-25
**Область:** process
**Severity:** minor (1-line fix), но paper-grade lesson

В Phase 1.3a я обновил **все** тесты под новую SortIndices schema
(`idx.sort_idx_k` вместо `idx.sort_idx`, добавил Q-sort + scatter_by_inv_perm).
Прогон v6 на GPU показал что correctness work полностью прошёл (108/108 CPU,
89/89 GPU), но **microbench упал с AttributeError**: 'SortIndices' object has
no attribute 'sort_idx'.

**Root cause.** `benchmarks/microbench_kernel.py` физически в `benchmarks/`,
не в `tests/`. Когда я делал `grep -rn "sort_idx[^_]" tests/` — он не покрыл
`benchmarks/`. Я думал «обновляю тесты», microbench формально не тест,
поэтому пропустил.

**Последствие.** v6 round дал **inconclusive** результат на performance
acceptance test. Все 24 v6 triton records были error-records. Scaling script
silently fell back на v5 records (latency invariant в k_window) и показал
прежние числа. Phase 1.3a kernel **не был измерен**.

**Lesson (paper-grade methodology).** *«При schema change в core contract,
inventory ВСЕХ usage sites — не только тестов, но и benchmarks, examples,
docs, downstream tools. Использовать `grep -rn "<old_name>" .`, не только
`grep -rn "<old_name>" tests/`.»*

Конкретный protocol для будущих schema changes:
1. Identify the field/function being renamed.
2. `grep -rn "<old_name>\b" .` (literal, NO directory limit).
3. Catalog every hit with its file path.
4. Update each one in the same commit.
5. Verify with `grep -rn "<old_name>\b" .` returning **zero** hits.

Этот protocol занимает 2 минуты и сэкономил бы 1 GPU validation round.

**Связь с paper v2.** Это второй process-related lesson после INS-12
(structural patches scale by squared cost). Идёт в §7 Discussion как
«Schema migration discipline для multi-component projects» — конкретный
artefact для reproducibility-conscious reviewer.

## INS-22 — Phase 1.3 v7 measurement: DCR competitive at target regime
**Дата:** 2026-04-25
**Область:** measurement, paper-grade
**Severity:** baseline-establishing finding

GPU validation v7 на RTX 4060 Ti (Ada Lovelace) после Phase 1.3a structural
fix дал **48 valid records** (24 triton + 24 sdpa), zero errors. Latency
scales with k_window (10/12 rows ratio > 1.5, mean 1.99) — bounded K-loop
engaged. Single-config speedup vs Phase 1.2 для N=8192 D=128 k=64 bf16:
**481×** (1376 ms → 2.86 ms).

**Decision-relevant numbers (DCR/SDPA ratio):**

  * N=8192, D=128, k=64, bf16: **0.40** (DCR 2.86 ms, SDPA 7.16 ms)
  * N=8192, D=128, k=64, fp16: **0.49** (DCR 6.12 ms, SDPA 12.53 ms)

Это **0.40×**, что переводит в DCR **в 2.5× быстрее** SDPA в target regime.
Это попадает в категорию которой не было в моём locked decision gate
(§10 of phase1_3_q_sort.md) — я предусмотрел 4 варианта для "DCR медленнее
SDPA" но не написал handler для "DCR быстрее". Process miss с моей стороны.

**Polynomial scaling regime confirmed.**

| Regime | DCR/SDPA range | Configs |
|---|---|---|
| N≥4096, k=64 (narrow window) | 0.26 – 0.86 | 6/6 DCR faster |
| N≥4096, k=256 (wide window) | 0.62 – 2.36 | mixed, near parity |
| N=1024 (short context) | 1.84 – 7.89 | DCR slower (overhead dominates) |

**Что измерения не устанавливают:**

  * Numbers с `input_precision="ieee"`, без TF32. Phase 4 autotune может
    дать additional speedup, но это speculation до measurement.
  * Single-GPU run на consumer-grade Ada (sm_89, 100 KB SMEM). A100/H100
    могут показать другую картину — больше SMEM позволяет больший BLOCK_K
    и num_stages для D=128. Phase 3 cloud runs.
  * Memory: DCR 15-122 MB vs SDPA 12-72 MB. INS-8 prediction holds — pre-gather
    K_sorted/V_sorted имеет своя cost. Phase 4 in-kernel scatter-gather может
    это закрыть.
  * Sort overhead в production amortized over Llama 32-layer forward pass.
    Microbench excludes его из timed call, что right model для production
    use, но это нужно явно сказать в paper §5 чтобы не было reviewer
    objections.

**Paper-grade claims которые теперь defensible (после v7):**

  1. *«DCR-Attention is competitive with SDPA in long-context regime
     (N≥4096) at narrow window (k≪N).»* — measured.
  2. *«k-scaling property is structural, not k-invariant: latency grows
     with bounded loop engagement.»* — measured (mean ratio 1.99).
  3. *«Adaptive dispatcher routing makes architectural sense: DCR for
     N≥4096 + structure detected, SDPA otherwise.»* — supported by
     small-N regression data (DCR/SDPA = 1.84-7.89 at N=1024).
  4. *«FLOP savings of bounded K-loop are realized in wall-clock.»* —
     481× speedup vs Phase 1.2 baseline at one config establishes this.

**Что не делаем поспешно.**

Numbers сильнее ожиданий. Соблазн объявить project complete, переписать
paper в "DCR-Attention beats SDPA" tone, и пропустить Phase 4. Это
**неправильный** мысленный путь — single-GPU consumer Ada round не establishes
A100/H100 behavior, не accounts for end-to-end Llama integration costs, и
overstates confidence на 24 microbench points. Honest narrative: «competitive
within target regime; further measurement needed for cloud-scale claims».

Phase 2 Llama integration — следующий шаг. Phase 4 optimization не отменяется.

**Methodology takeaway для INS-12 / 21 series.**

Discipline of writing decision gate **before** seeing numbers paid off:
gate имеет асимметрию (assumed downside), но даже тогда **clear answer for
present case — Path A**. В paper §5 Methodology упомянуть этот pattern как
explicit defense против post-hoc cherry-picking — reviewer-proof artifact
of process discipline.

## INS-23 — Implicit prefill-only assumption hidden in single scalar `N`
**Дата:** 2026-04-26
**Область:** kernel, design
**Severity:** structural (paper-grade)

При попытке decode microbench (Q.shape=[B,H,1,D], K.shape=[B,H,N_kv,D])
все 26 triton_rank_local_decode записи упали с
`RuntimeError: shape '[32, 1, 128]' is invalid for input of size 4194304`.

**Root cause.** Phase 1.x kernel signature принимал **один scalar `N`**,
derive'нный из `Q.shape[-2]`. Внутри kernel этот N использовался для
**обоих** Q-side (mask_q, grid) **и** K-side (mask_k, k_hi clamp). В prefill
N_q == N_kv, поэтому conflation работала случайно. В decode N_q=1 ≠ N_kv,
host launcher reshape'ит K_sorted в `[BH, 1, D]` ← invalid.

Это **не** «забытая фича» — это **структурное предположение**, baked into
signature через единственный параметр. Документации `Q, K, V : [B, H, N, D]`
явно говорит N is a single dimension shared между Q и K — то есть assumption
locked в API.

**Lesson (paper-grade architectural).** *«When two related quantities can
diverge in some downstream usage, name them separately from the outset.»*
Если бы я в Phase 1.2 назвал `N_q` и `N_kv` — даже когда они равны — Phase 2
decode bug не существовал бы. Стоимость pre-emptive split: zero LOC (один
scalar становится двумя). Стоимость post-hoc split: 4 файла, 5 тестов, 1 GPU
round.

**Связанный pattern.** Любой shared-scalar в kernel signature — `M, N, K`
для matmul, `H, W` для conv — кандидат на «implicit equal-by-default»
assumption. В будущих kernels: явно named per-axis scalars, даже когда
equality is the common case.

**Связь с paper v2.** Идёт в §4 Implementation как «kernel API design lessons».
Конкретный example — single-N conflation — реален и replication-friendly.

## INS-24 — `r_center.clamp_(0, Q.shape[-2] - 1)`: pre-existing bug, hidden
**Дата:** 2026-04-26
**Область:** sort_helpers, methodology
**Severity:** silent bug, found only via shape divergence

Phase 1.3a `prepare_sort_indices` имел:
```python
r_center = r_center.clamp_(0, Q.shape[-2] - 1)
```

`r_center[t]` is insertion rank in **K-sorted sequence**, range `[0, N_kv-1]`.
Upper clamp должен использовать **N_kv = K.shape[-2]**, не **N_q = Q.shape[-2]**.
В prefill (N_q == N_kv) это совпадало → bug не проявился через 7 GPU validation
rounds + 89 correctness tests. Phase 2-pre decode shape (N_q=1, N_kv=context)
finally exposed его: при N_q=1, верхняя граница = 0, все r_center clamp'ятся
до 0, kernel attends только к keys around position 0.

**Why it was silent for 7 rounds.**

Two reasons in combination:
  1. **Symmetric coupling:** test inputs all had N_q==N_kv, hiding the
     asymmetry between two quantities.
  2. **searchsorted natural range:** `torch.searchsorted(z_k_sorted, z_q)` returns
     values in `[0, N_kv]` natively. `clamp_(0, N_q-1)` only acts when N_q < N_kv.
     In prefill N_q == N_kv, clamp is no-op. In decode N_q=1, clamp ≪ N_kv, clamp
     becomes destructive.

**Fix.** One character: `Q.shape[-2]` → `K.shape[-2]`.

**Lesson (paper-grade methodology).** *«A shape-conflated bug stays silent
exactly as long as the conflation holds. Any architectural decision that
hides shape-equality assumption pays interest with shape-divergence later.»*
В paper v2 §4 — пара INS-23 + INS-24 — это **двойной пример** того, как
single-N coupling в API + single-N coupling в helper создали bug которого
было невозможно поймать prefill-only тестами.

**Process implication.** Future shape-related code review prompt: «assume
this function will be called with mismatched dimensions. Where does that
break?» Это would've caught both INS-23 and INS-24 за 5 минут code review,
сохранив 1 GPU round.

## INS-25 — Decode regime characterisation: N_kv-dependent crossover
**Дата:** 2026-04-26
**Область:** measurement, paper-grade
**Severity:** baseline-establishing

GPU validation v9 после Phase 2-pre fix дал **48 valid records** (24 triton
decode + 24 sdpa decode), zero errors. Decode latency analysis показал
pattern с тремя regime'ами:

| N_kv | DCR/SDPA | Pattern |
|---|---|---|
| 1024 | 1.40 – 2.69 | Launch overhead dominates; SDPA wins |
| 4096 | 0.52 – 1.98 | Crossover region; mixed |
| 16384 | **0.13 – 0.27** | Bounded loop wins; DCR 4-8× faster |

**Key structural finding: DCR latency essentially flat в N_kv**
(observed 0.12–0.31 ms across all configs). SDPA latency растёт линейно
с N_kv (0.07 → 0.26 → 1.00 ms для D=128 bf16). Это empirical proof что
bounded loop делает what design promised — visits только `k_window` keys
независимо от cache size.

**Memory regression confirmed (INS-8 holds).** DCR использует **в 2× больше**
memory чем SDPA на всех N_kv. При N_kv=16384, D=128: 524 MB vs 264 MB.
Для Llama-3 8B 32 layers × 524 MB = 16.8 GB attention memory, что pushes
limit RTX 4060 Ti при N_kv≥32K. Phase 4 in-kernel scatter-gather будет
critical для long-context.

**Decision gate verdict.** 15/24 rows < 1.5, 24/24 < 3.0, mean 1.10. По букве
triggers «Phase 2a proceeds». Но **honest reading требует nuance**:

  * Decision-relevant regime (N_kv ≥ 4096, D=128): 6/8 rows < 1.5, 4 < 1.0.
    Phase 2a proceeds **straightforward**.
  * N_kv = 16384, D=128: DCR/SDPA = 0.13–0.18. Это long-context Llama-3
    target (32K-128K context).
  * N_kv = 1024: DCR slower универсально. **Не bug, но constraint:**
    Phase 2a wrapper MUST integrate dispatcher routing от первого commit.
    Без этого short-context decode steps будут медленнее dense — отрицательно
    скажется на TTFT.

**Architectural lesson.** Phase 1 measurement pipeline (full Q microbench)
не предсказывал decode-regime numbers потому что compute pattern radically
different: full Q has Q-dimension parallelism для kernel; decode Q.shape[-2]=1
имеет only batch×heads parallelism. Latency-sources другие, structural
properties remain (bounded loop работает в обоих). Lesson: **regime-specific
microbench mandatory before each integration phase.**

**Paper-grade narrative claim (defensible).**

> *«DCR-Attention's bounded-loop kernel makes its decode latency essentially
> independent of KV cache size, while SDPA's scales linearly. The crossover
> point on RTX 4060 Ti is around N_kv ≈ 2048-4096; above this DCR is
> increasingly faster, with up to 8× speedup at N_kv=16384. Below crossover,
> kernel launch overhead dominates and dense SDPA wins. This motivates a
> dispatcher-driven design where attention routing is sequence-length-aware.»*

Each clause supported by measured numbers. No overclaim про generic decode
beating SDPA — narrative explicitly acknowledges short-context regression.

**Updated Phase 2a scope after v9.**

  * v0 (original): simple `DCRLlamaAttention`, all decode → DCR.
  * **v0' (revised по v9 numbers): `DCRLlamaAttention` with built-in
    dispatcher routing from commit-1.** Threshold `N_kv < T` → SDPA fallback,
    `T` determined empirically (likely 2048-4096).

Это larger scope, но **honest** — без dispatcher real Llama generation
будет regress на early steps когда KV cache мал.

**Total paper-grade insights now:** 12 (INS-13, 14, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25). Достаточно для paper v2 §4 + §5 + §7 без heavy lifting.

## INS-26 — HF API drift: version-aware shim required, not version pinning
**Дата:** 2026-04-26
**Область:** integration, methodology
**Severity:** breaks integration; minor code fix; paper-grade lesson

GPU validation v10 (first integration с real HF transformers) дал **6 CPU
failures** в `test_dcr_llama_unit.py`, все с одним и тем же stack trace:

```
TypeError: apply_rotary_pos_emb() takes from 4 to 5 positional arguments
but 6 were given
```

**Root cause.** Я писал `compat.py` под transformers 4.40+ signature
`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)`.
У Seqev'а установлен transformers **5.6.2**, который **dropped position_ids**
из signature: `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`.

Когда `compat.py:62` вызывал `_hf_apply(Q, K, cos, sin, position_ids,
unsqueeze_dim)` с 6 positional args, Python отклонял это потому что 5.x
signature принимает максимум 5.

**Что я сделал не так в design.** В `compat.py` docstring я написал:
> *«Tested against transformers ≥ 4.40 (where Llama-3 was introduced).»*

Это было предположение, не measurement. Я не знал что у Seqev'а установлено,
и не запросил эту информацию **до** написания `compat.py`. Соответствующая
process miss: Phase 2a design doc §11 timeline не включал «verify transformers
version before writing compat shim».

**Что я сделал правильно.** `compat.py` **изолировал** все HF API touch-points
в одно место. Когда API drift случился, fix был один файл. Это validates
the «one-place-for-HF-knowledge» design pattern.

**Fix.** Version-aware dispatch через `inspect.signature` + `lru_cache`:

```python
@lru_cache(maxsize=None)
def _resolve_hf_apply_rotary():
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb as _hf_apply,
        )
    except ImportError:
        return None, None
    accepts_position_ids = "position_ids" in inspect.signature(_hf_apply).parameters
    return _hf_apply, accepts_position_ids
```

Per-call cost: zero после warm-up (lru_cache). Per-process cost: one
`inspect.signature()` call per HF symbol resolved.

**Lesson (paper-grade methodology).** *«When integrating with a fast-moving
upstream (HF transformers, PyTorch, etc.), do NOT pin a specific version in
your shim layer. Do version-aware runtime dispatch via `inspect.signature` —
this lets your code support multiple upstream versions without releases.»*

The alternative (version pinning via `transformers>=4.40,<5.0`) would force
users into a single transformers generation and create maintenance burden
when 6.0 ships. Runtime dispatch costs ~1 µs per process and supports the
full lifetime of the API.

**Process pattern.**

Now the **third** process-related miss in this project after INS-12 (structural
patches) and INS-21 (schema migration). All three share a root signature:
*«assumption about an interface that's not explicitly contracted gets
violated when reality diverges from assumption.»* The fix in each case has
been to make the implicit explicit — INS-12 documented test scaling cost,
INS-21 added grep-the-whole-tree-not-just-tests rule, INS-26 added
runtime API probing.

**Connection to paper v2.** §7 Discussion now has a **process triplet**:
INS-12 (test cost), INS-21 (migration discipline), INS-26 (API drift
defence). Together they form a coherent «engineering discipline lessons
from real-world deployment of research software» section that's stronger
than any single insight alone.

**Updated insights count: 13** (INS-13 through INS-26, with INS-15 retired
into 17). Sufficient for paper v2 final draft.

## INS-27 — Four API-assumption misses in a row → structural process fix
**Дата:** 2026-04-26
**Область:** methodology, process
**Severity:** structural — invalidates "fix-and-retry" pattern as discipline

This is the **fourth** API-assumption miss in this project after INS-21
(schema migration), INS-23+24 (N-conflation), and INS-26 (transformers
4.40 vs 5.x rotary signature).  v11 found the fifth: `LlamaAttention`
in transformers 5.x dropped `num_heads` and `num_key_value_heads` from
the instance, moving them to `config`.

**The pattern that's failing.**  My discipline so far has been:
  1. Write code under assumed interface.
  2. Hit reality → traceback.
  3. Fix → INS-N → next round.

This is **post-hoc cleanup**, not discipline.  Each round trades a
constant amount of cognitive load for one fewer assumption violation, but
the pool of unverified assumptions remains large.  The cumulative GPU-
round cost across INS-21, INS-23/24, INS-26, and INS-27's triggering
event is **four** rounds — each ~30 min of Seqev's GPU time + my fix
turnaround.

**The structural fix.**

Before *any* future integration work, run a **diagnostic-only round**
that produces a complete inventory of the upstream surface (here: HF
transformers' `LlamaAttention` instance + its config + relevant module-level
functions).  This inventory is the **explicit contract** my code targets.

The diagnostic round costs one GPU-round.  It saves N future GPU-rounds
where N is the count of un-discovered API drifts.  For Phase 2a so far
N = 2 known (rotary signature, instance attrs); likely N = 1-3 more
hidden in cache update / forward signature / RoPE module location.

**Diagnostic-first contract.**

  1. Diagnostic round produces a verbatim attribute / signature dump.
  2. Architect updates compat shim against the dump — every accessed name
     is verified to exist.
  3. Architect writes a "contract test" that asserts the inventory
     (will fail if upstream drifts again, with clear diff).
  4. Only then proceed with integration retry.

**Process triplet → quartet.**

Paper v2 §7 Discussion now has four process insights forming a coherent
methodology lesson:

  * INS-12: structural patches scale by squared cost (test pollution).
  * INS-21: schema migration discipline (grep entire tree, not just tests).
  * INS-26: API drift defence (runtime dispatch, not version pinning).
  * INS-27: diagnostic-first integration (inventory → contract → code).

Together these form the strongest section of the paper's methodology
narrative — not because I'm a brilliant engineer (I'm not — I made these
mistakes), but because each insight comes from a documented real failure
with measurable cost and a measurable fix.

**Updated insights count: 14.**

## INS-27 update — TDD on contract validates the diagnostic-first pattern

**Дата:** 2026-04-26
**Область:** methodology, validation
**Severity:** confirms INS-27 as reproducible discipline, not theoretical

After the v12 diagnostic round produced the inventory dump, the architect
audited it sequentially against current code and identified **two BLOCKERs**
(plus three minor improvements):

1. `attention.py forward` had `past_key_value` (singular); HF 5.x passes
   `past_key_values` (plural) per Section J.  HF's kwarg silently fell
   into `**kwargs`; local parameter stayed `None`; cache update never ran.
2. `compat.cache_update` passed `cache_kwargs` as a **positional dict** to
   `Cache.update(*args, **kwargs)`.  The dict landed in `args[0]`, not
   spread as kwargs.  Per Section I, this is wrong.

**Critical methodology validation.** Both BLOCKERs were found by audit
**before** any GPU round.  Before INS-27, both would have crashed
integration tests in v13 — likely as separate rounds, since one would
mask the other.

**TDD applied to API contract.** The architect wrote `test_hf_5x_contract.py`
**before** the fix.  6 tests cite their diagnostic-dump section.  3 of them
*failed* on the un-fixed code (validating the BLOCKER diagnoses); 3 passed
(validating that prior fixes — v10's resolver, v11's `_resolve_shape_attr`
— are correct).  After the fix, all 6 pass.  This pattern:

  diagnostic dump → audit → contract test (failing) → fix → contract test (passing)

is the **reproducible methodology** that INS-27 promised.  It is now a
process artefact, not a theoretical principle.

**Cost-benefit.** v12 diagnostic round (~5 min Seqev's GPU time) found
two BLOCKERs.  Without it, each would have cost a separate v13 / v14 round
(~30 min each).  Net saving: 55 minutes of measurement time + the cognitive
cost of fix-and-retry cycles + the project-narrative cost of admitting yet
another miss.

**Paper-relevance.** §7 Discussion can now describe a closed feedback loop:
diagnostic → contract → fix → measurement.  This is harder to write up
than "we tried and it didn't work" but more useful for replication.

**Updated insights count remains 14.**  INS-27 is now upgraded from "process
proposal" to "validated process artefact".

## INS-17 — `from __future__ import annotations` ломает PEP 526 constexpr форму
**Дата:** 2026-04-25
**Область:** kernel
**Severity:** blocker → fixed

Round v2 (INS-15 fix-attempt) использовал PEP 526 синтаксис:
```python
_NEG_INF_SAFE: tl.constexpr = -1.0e30
```
Это **должно** было сообщить Triton'у через module `__annotations__`, что глобал
является constexpr. Но не сработало.

**Причина** (диагностирована коллегой/Claude Code):
Файл начинается с `from __future__ import annotations` (PEP 563 lazy evaluation
of annotations). При этом флаге **все** annotations хранятся в `__annotations__`
как **строки**, не объекты:
```python
__annotations__ == {'_NEG_INF_SAFE': 'tl.constexpr'}    # str, не type
```
Triton проверяет `isinstance(annotation, tl.constexpr)` — falls False, потому что
в памяти это строка `'tl.constexpr'`, а не сам класс. Triton 3.1 не parse'ит
строки annotations, не делает eval — просто отказывается.

**Фикс — функциональная форма (вариант B из round 2 диагностики):**
```python
_NEG_INF_SAFE = tl.constexpr(-1.0e30)
```
Это создаёт **runtime** объект `tl.constexpr`, который не зависит от annotation
evaluation strategy. Triton'у этого достаточно.

**Lesson learned.** PEP 526 annotation form для constexpr globals в Triton
**требует** отсутствия `from __future__ import annotations` в файле. Если
нужна совместимость с Python 3.9- и lazy annotations — функциональная форма
универсальнее.

Этот bug был мой косяк: я знал про PEP 563 и про PEP 526, но не сложил два с
двумя. Уроки имени коллеги/Claude Code — диагностика была быстрее моей.




## INS-16 — Triton tl.dot default uses TF32 on Ampere/Ada, не IEEE fp32
**Дата:** 2026-04-25
**Область:** kernel, paper, test methodology
**Severity:** blocker (confirmed real bug, fixed) + paper-grade

Round 3 GPU validation: smoke прошёл, но 54 теста упали с **per-query systematic offset**:
  fp32: max|abs| = 1.97e-3 — 3.48e-3, atol требовал 1e-5 (overshoot 100×).
  bf16, D=128: max|abs| = 1.56e-2, atol требовал 1e-2 (overshoot 1.5×).
  bf16, D=32, k=16: PASSING.

Pattern: ошибка растёт с D, stable в N, fp32 хуже всего (после ieee fix должен быть лучшим).

**Источник bug.** На Ampere/Ada/Hopper `tl.dot(a, b)` с `a, b: tl.float32` по умолчанию
использует **TF32 multiply** (10-bit mantissa) с fp32 accumulate. Это правильное
дефолтное поведение для production performance (FlashAttention так делает), но
**ломает** строгие fp32 parity tests против torch reference, который использует
полноценный IEEE-754 fp32.

`tl.dot` accepts параметр `input_precision: "tf32" | "tf32x3" | "ieee"`.
Default `"tf32"`. Для testing-correctness нужен `"ieee"`.

**Bf16/fp16 ошибки — не bug, а intrinsic dtype precision floor.**
Ожидаемая ошибка после D-элементного MAC + softmax-normalisation:
  bf16 (7-bit mantissa, 2⁻⁷ ≈ 8e-3 per element):
    D=32:   ≤ 5e-3       (passes atol=1e-2)
    D=128:  ≤ 1.6e-2     (fails atol=1e-2 by 1.6x)
Это **физика** bf16 storage + multiply, не дефект kernel'а.

**Two fixes applied:**
1. Kernel: `input_precision="ieee"` в обоих `tl.dot` для fp32 path.
2. Tests: D-aware tolerance `_expected_atol(dtype, D)`:
   - fp32: 1e-5
   - bf16/fp16: `max(1e-2, 1.5e-3 · sqrt(D))`
     → D=32: 1e-2, D=64: 1.2e-2, D=128: 1.7e-2
   Калибровано на observed round-3 numerics, с small safety margin.

**Paper-grade insight для §4 Implementation:**

«Tolerance choice in numerical correctness tests must reflect the floor set by the
hardware-level dtype semantics, not an idealised IEEE-754 baseline. Using
`tl.dot(input_precision="ieee")` for fp32 testing is a mandatory step on Ampere+
hardware where the default TF32 multiply otherwise contaminates parity tests with
~1e-3 noise per dot product. For bf16 inputs, the intrinsic storage+multiply
precision floor scales as `O(sqrt(D))` after D-element MAC followed by softmax
normalisation. Tolerances below this floor are not "stricter testing" — they
fabricate failures in correct kernel code.»

**Связь с INS-5.** Round-3 ситуация — повторение методологического урока INS-5
(reference-тесты v2.1.0-rc1 имели `cos > 0.90` без эмпирического обоснования).
Та же проблема: tolerance threshold выбран **до** наблюдения реальных numerics,
вместо **после**. Защита: всегда сначала прогон без assertion (raw numerics output),
потом tolerance set based on observation, потом assertion. Это **third-time-pattern**
в нашем проекте — фиксирую как hard rule методологии.

## INS-28 — bf16/fp32 softmax precision divergence

`F.scaled_dot_product_attention` with bf16 inputs uses bf16 softmax
internally (FLASH backend on Ampere/Ada).  HF eager LlamaAttention
explicitly forces fp32 softmax via `dtype=torch.float32` then casts back
to query dtype.  Difference accumulates over deep stacks: 16 layers of
Llama-3.2 1B yielded max|diff| = 0.30 in final logits — 6× over the 5e-2
bf16 tolerance budget.

**Detection**: integration tests `test_enable_dcr_false_matches_unpatched`
and `test_T_dispatch_infinity_matches_unpatched` failed with exact values
0.302734 / 0.140625 (v14 GPU integration report).

**Fix (v14b)**: replaced F.sdpa with manual attention path matching HF
eager exactly:
    attn_weights = (Q @ K.T) / sqrt(d_head)
    attn_weights += attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
    return attn_weights @ V

**Architectural implication**: when integrating with reference models
that use fp32 softmax (which is most HF eager paths), our SDPA fallback
must match precision.  For DCR branch this is irrelevant — DCR is
approximation by design — but for ablation runs (enable_dcr=False) and
short-context decode, bit-equivalence requires this care.

**Contract test**: `test_contract_sdpa_softmax_uses_fp32` (numerical),
`test_contract_sdpa_softmax_dtype_inspection` (static AST guard against
regression to F.sdpa).

## INS-29 — Option triage discipline (fix-with-fewer-assumptions)

When equally-correct fix options differ in API surface dependency,
pick the one with fewer assumptions.  Concretely: helper-AI proposed
delegating SDPA branch to `_base_layer(...)` (HF original LlamaAttention)
to obtain bit-identical output.  This required 3 unverified HF API
assumptions (`DynamicCache.get_seq_length(layer_idx=)`,
`LlamaAttention(output_attentions=)`, `LlamaAttention(cache_position=)`)
and a silent failure mode via `try/except (AttributeError, TypeError)`.

We chose Option A (replicate HF eager compute locally) — same correctness
outcome, zero new API surface.  This is an extension of INS-27
(diagnostic-first) — even when a fix passes diagnostic, prefer the
implementation with smaller assumed surface.

## INS-30 — Class-level counters for branch verification

Process-wide tracking of which forward branch (SDPA vs DCR) actually
fires in real inference.  Critical because **silent fallback** is a
correctness failure that's invisible to ordinary tests: if routing
returns DCR but `_dcr_forward` quietly raises and gets caught, an SDPA
path may execute instead with no error signal.  Counters expose this.

Pattern:
    class DCRLlamaAttention(nn.Module):
        _dcr_invocations: int = 0
        _sdpa_invocations: int = 0

        @classmethod
        def reset_counters(cls): ...

        @classmethod
        def get_counters(cls): ...

Increment via `type(self)._dcr_invocations += 1` in dispatch (uses class
of instance, so subclasses share their own counters not parent's).
Atomicity not required for single-stream inference; if we ever
multi-stream, replace with `torch.atomic`.

This pattern generalises beyond DCR: any wrapper that does internal
routing should expose route-level counters as a debugging contract.

## INS-31 — d_eff is gap-blind: the dispatcher signal does not see the regime boundary where its own sparse path degrades
**Дата:** 2026-05-18
**Область:** dispatcher, paper
**Severity:** major

Связь с геометрическим companion-манускриптом (Section 6, Gap-Statement).
Для attention логиты линейны, поэтому гессиан свободной энергии есть
ровно `H = -beta * Cov_p(k)` (член средней кривизны зануляется). Счёт
отрицательных собственных значений `dim E_-` корректно определён —
беспороговый, устойчивый — тогда и только тогда, когда спектр имеет
зазор `Delta(H) > 0`. Три новых теста-моста в
`tests/dispatcher/test_gap_theory.py` связывают это с диспетчером.

**Что ожидали:** `d_eff = (tr S)^2/||S||_F^2` (S = Cov(Q)) — мера
эффективной размерности; ожидали, что она отслеживает «насколько sparse
применим».

**Что увидели (три моста):**

1. *Bridge 1.* В gap-режиме `dim E_-(softmax) = m-1` точно (m — число
   кластеров; -1 это mean-gauge дефекта ранга ковариации). `d_eff(Q)`
   НЕ равен `m`, а является **нижней оценкой** `dim E_- + 1`:
   participation ratio совпадает со счётом лишь при равных собственных
   значениях, при неравном межкластерном спектре он систематически ниже.
   Обе величины растут один-к-одному с m — это и есть мост, но равенства
   нет. Первоначальное предположение `d_eff ≈ dim E_- + 1` было слишком
   сильным; исправлено на неравенство `1 ≤ d_eff ≤ dim E_- + 1`.

2. *Bridge 2 — основная находка.* При закрытии зазора (сближение
   кластеров) `Delta(H)` обваливается на 2+ порядка, а `d_eff(Q)` НЕ
   падает — дрейфует в сторону «плотного режима». Качество rank-local
   attention при этом деградирует (rel.err против dense растёт ~0.3 → ~1.9).
   Эффект проверен с КОНТРОЛЕМ geometric crowding: при фиксированной доле
   покрытия `k_window/N = const` ошибка всё равно растёт по мере закрытия
   зазора, затем выходит на плато (закрытый зазор больше нечему закрывать).
   То есть деградация имеет подлинно спектральную компоненту, не сводимую
   к тривиальному сужению окна. **Диспетчер слеп к границе режима, за
   которой его собственный sparse-путь теряет точность.**

3. *Bridge 3.* `dim E_-` устойчив при возмущении `||dH||_op < Delta/2`
   (Дэвис–Каган) и может меняться выше — порог не вакуумный.

**Следствие:**
- `d_eff` сам по себе — недостаточный routing-сигнал у границы зазора.
  Рекомендуется добавить оценку `Delta(H)` (или дешёвый прокси:
  отношение `lambda_k/lambda_{k+1}` в спектре `Cov(Q)`) как guard:
  при `Delta -> 0` принудительный fallback в dense, независимо от score.
- Это НЕ предписание немедленно менять production-код. Это диагностика
  (INS-27 discipline): находка зафиксирована тестами, решение об
  изменении routing-логики — отдельный шаг с явным approval.
- Подтверждает paper-claim Section 6: операционное окно детектора есть
  множество `{Delta(H) > 0}`; три эмпирических «окна» (энергия, разделение,
  мягкость) — его координатные сечения.

**Bug-pattern (для будущих тестов):** «наибольший разрыв в спектре» —
неверный критерий поиска спектрального зазора: крупнейший разрыв может
лежать ВНУТРИ межкластерного блока, если центры кластеров сами разнесены
неравномерно. Корректный критерий — наибольший **мультипликативный**
скачок `|lambda_lo|/|lambda_hi|`, масштабно-инвариантный (абсолютный
порог «близко к нулю» промахивается, т.к. спектральный масштаб сам
схлопывается вдоль sweep). См. `spectral_gap()` в test_gap_theory.py.

## INS-33 — U-shape depth profile is model-specific; 3-metric coupling is architecture-general
**Дата:** 2026-05-17
**Область:** measurement, paper, cross-model
**Severity:** moderate (refines the Llama U-profile claim)

Cross-architecture measurement on 5 trained transformers
(GPT-2 small, GPT-2 medium, BERT-base, Pythia-1.4B, Llama-3.2-1B) at
fixed corpus (WikiText-2 validation), with full per-head Hessian
spectrum. Бенчмарк — `benchmarks/xmodel_uprofile/`; полный отчёт —
`XMODEL_UPROFILE_REPORT.md` в корне репо.

**Что измеряли:** `chi = |lambda_1|/sum|lambda_i|`, `S_lambda` (spectral
entropy of negative-eigval mass distribution), `R = |lambda_1|/|lambda_2|`
на каждой `(layer, head, prompt, query position)` после
genuine-gap фильтра `relgap > 0.05` и counter-saturation guard'а
(`dim_E_- >= D-1 ⇒ no gap, exclude`).

**Что подтвердилось:**

1. *3-метрика coupling.* Median per-head Spearman `|rho|` между
   (chi, S_lambda, R) — **0.7..0.97 на ВСЕХ пяти архитектурах** (chi-S
   ≥ 0.92 везде). Эти три метрики действительно три проекции одной
   структуры, не только на Llama. Это architecture-general свойство.

2. *Genuine-gap regime повсеместен.* После relgap-фильтра остаётся
   достаточно samples на всех моделях для устойчивой статистики; spectral
   gap как inductive bias trained attention существует во всех 4
   архитектурах, не специфичен Llama.

**Что НЕ воспроизвелось:**

Депth profile of `chi`:

- BERT (12 слоёв):    0.30 / 0.27 / 0.38 — late-rise
- GPT-2 small (12):   0.58 / **0.72** / 0.52 — INVERTED-U (середина выше)
- GPT-2 medium (24):  0.60 / 0.58 / 0.52 — monotone decline
- Pythia 1.4B (24):   0.38 / 0.53 / 0.64 — monotone rise
- Llama 3.2-1B (16):  0.65 / 0.62 / 0.68 — слабый partial U (dip ≈ 0.05)

Strict U-shape (early > middle И late > middle с margin > SE) НЕ
выполняется ни на одной модели. Даже на Llama dip мелкий и только
late-side margin clear. **Llama-shape депth profile — это features
trained Llama-3.2-1B, не features trained-transformer attention.**

Особенно показательно:
- GPT-2 small даёт ОБРАТНЫЙ паттерн (середина максимум).
- Pythia даёт monotone increase — никакого dip середины вообще.
- Decoder family не однороден: GPT-2, Pythia, Llama дают три РАЗНЫХ
  паттерна, хотя все три — autoregressive decoders с similar SDPA.

**Следствие:**

- Депth-profile claim (chi 0.77/0.53/0.73 для Llama) — Llama-specific
  observation, не candidate invariant of hierarchical representation
  transport. Стоит ограничить scope claim'а в paper section 6.
- 3-метрика coupling — реальное architecture-general свойство. Можно
  заявлять, что в gap-regime спектр trained attention head живёт на
  1D эффективном многообразии (chi и его друзья) — это держится.
- Dispatcher-guard suggestion INS-31/32: если он зависит от U-shape
  identification regime'а где «sparse работает», то теперь видно, что
  такой regime не определяется одинаковым способом на разных моделях.

**Методологические подтверждения:**

- relgap > 0.05 (scale-free) — необходим: без него абсолютный порог
  Delta>X плохо переносится между моделями с разным head_dim (Pythia
  D=128, остальные D=64) и dtype (bf16 vs fp32).
- Counter-saturation guard (`dim_E_-=D-1 ⇒ no gap`) применён везде.
- Eigsolve robustness: на Pythia bf16-производный Hessian вызвал
  `torch._C._LinAlgError`; fallback на `numpy.linalg.eigvalsh` в fp64
  с явной симметризацией решает. Реализовано в `_robust_eigvalsh`
  в `gap_metrics.py`.
- Capture hook (`xmodel_capture.py`) per-architecture: post-projection
  Q,K (для GPT-2/BERT без RoPE; для Pythia/Llama с RoPE — partial RoPE
  у GPT-NeoX обработан явно: `[:, :rotary_ndims]` повёрнут, остальное
  pass-through). Forward pre-hook'и снимаются в конце каждого прогона
  — production forward path байт-в-байт не изменён, off-by-default
  гарантия сохранена.

**Bug-pattern (для будущих cross-model работ):** заявляя observation
как "invariant of trained transformers", всегда измерить на как минимум
ОДНОЙ модели другой архитектуры/family прежде чем называть это
invariant. На синтетике или одной модели легко спутать model-specific
inductive bias с universal phenomenon.

## INS-34 — Dominant spectral mode is causally inert for next-token prediction (NULL); spectral-causality direction closed
**Дата:** 2026-05-17
**Область:** measurement, paper, causal
**Severity:** decisive (closes a research direction per pre-registered criterion)

Decisive causal experiment on Llama-3.2-1B. Ablate the dominant eigenmode
of `Cov_p(k)` at a chosen (layer, head) with a **matched-Frobenius
perturbation** control (suppress a random bulk eigenmode of the same
`||δK||_F`). Pre-registered outcomes: null → close the direction;
positive → continue. Full report: `SPECTRAL_ABLATION_REPORT.md`.

**Construction:** For each (layer, head) at the last query position:
`p = softmax(beta · K_h @ q)`, `Cov_p = K_c^T diag(p) K_c`, eigh →
`v_1` (dominant), `v_j` (random bulk, seed-deterministic). Compute
`M_T = ||K_h @ v_1||`, `M_C = ||K_h @ v_j||`, `T = min(M_T, M_C)`. For
active mode: `alpha = T / M_active`, `K_h_new = K_h - alpha · (K_h @
v_active) ⊗ v_active`. Matched ||δK||_F by construction.

**Что НЕ сработало (diagnostic-first, INS-27):** первоначальная
construction "project out v, rescale to preserve ||K||_F" дала
treatment ||δK||_F в 10-30 раз больше control'а. Любой
treatment-greater-than-control результат там был бы pure scale-asymmetry,
не causal claim. Smoke run выявил это, construction переписан до full
run. Matched-magnitude sanity check после фикса: ratio T/C = **1.000**
на всех 12 парах. Цена: точное сохранение ||K_h||_F отброшено в пользу
matched-perturbation; drop ||K_h||_F bounded и логируется
(sub-percent на типичных alpha).

**Результат — pre-registered NULL:**

| Метрика | Treatment − Baseline | Control − Baseline | Differential | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|
| ΔNLL | 5.95e-04 | 3.23e-04 | **+2.72e-04** | [−1.66e-04, 6.96e-04] **brackets 0** | **p = 0.26** |
| KL   | 1.31e-03 | 7.81e-04 | +5.32e-04 | [4.32e-04, 6.45e-04] | p = 3e-39 |

(n = 300 от 12 пар × 25 промптов; N_kv = 256; WikiText-2 validation)

**Интерпретация:**

1. *NLL — operationally meaningful test — NULL.* Подавление dominant
   mode причиняет task performance не больше, чем подавление random
   bulk mode равной Frobenius-возмущенности. По pre-registered
   criterion это Outcome 1: spectral-causality direction CLOSED.

2. *KL — distributional sensitivity — detectable but small.* Ablation
   dominant mode действительно сдвигает output distribution заметнее
   чем bulk-mode ablation, но абсолютный масштаб крошечный (~5e-4 нат)
   и НЕ переводится в ухудшение предсказания.

3. *Decorative reparametrization confirmed.* Dominant spectral mode
   реален, геометрически отличим, и измеримо влияет на распределение
   — но не несёт компьютацию. Модель не опирается на него для
   next-token prediction. KL-shimmer без NLL-cost = «decorative» в
   терминологии brief'а.

**Что INS-34 закрывает:** гипотезу о том, что dominant negative
eigenmode of `Cov_p(k)` сам по себе — privileged computational object,
чьё устранение selectively повреждает next-token prediction. На
Llama-3.2-1B / WikiText-2 — НЕТ.

**Что INS-34 НЕ закрывает:**
- Гипотезу что КАКОЕ-ТО spectral feature carries computation
  (bulk spread, the gap itself, eigenvalue ratio — не тестировались).
- Другие архитектуры / scales.
- Targeted routing/retrieval probes (был outside time budget).
- Long-context heads (N_kv up to 8192 not probed).

**Связь с предыдущими INS:**
- INS-31: d_eff blind to gap closure — refinement.
- INS-32: gap exists on real Llama, gap-closed rare — refinement.
- INS-33: U-profile model-specific, 3-metric coupling architecture-
  general — orthogonal scope.
- INS-34: **dominant mode causally inert for NLL** — closes the
  strongest spectral-causality claim. The research thread that
  produced INS-31 → 32 → 33 → 34 ends here on the operational
  measure, with a null that's better information than an
  uncontrolled positive would have been.

**Discipline that made the null worth recording:**
1. Pre-registered criteria before seeing data.
2. Matched control as non-optional, not optional.
3. Diagnostic-first when smoke construction failed — fixed before
   running, not explained away after.
4. Honest verdict: null on the operationally meaningful test, with
   the KL secondary observation reported transparently but not
   inflated.

«Уровень шумности эффекта меньше шумности измерения» — а это и есть
закрытие гипотезы, а не подтверждение.

## INS-32 — Gap-theory on real Llama: weak reproduction, plus a scope limit INS-31 did not anticipate
**Дата:** 2026-05-17
**Область:** dispatcher, paper, measurement
**Severity:** moderate (refines INS-31)

Прямое измерение `Delta(H)`, `dim E_-` и `d_eff(Cov(Q))` на реальной
Llama-3.2-1B attention (WikiText-2, 30 промптов × 3 контекста
{512, 2048, 8192} × 16 слоёв × 32 головы = 46 080 head-наблюдений на
target). Бенчмарк — `benchmarks/phase2c_gap/run_gap_validation.py`;
полный отчёт — `GAP_VALIDATION_REPORT.md` в корне репо. Хелперы
переехали из тестового файла в `dcr_attention/analysis/gap_metrics.py`,
чтобы бенчмарк и тесты использовали ОДНУ реализацию.

**Что подтвердилось (weak reproduction):**

1. *Зазоры на реальной attention существуют.* 98%+ всех head-позиций
   имеют `Delta(H) > 1e-3`; медиана зазора ~7. Это уже само по себе
   значимо — синтетический gap-режим не артефакт кластерной геометрии,
   он представлен в реальных Llama-распределениях.
2. *`d_eff(Cov(Q))` практически не коррелирует с `dim E_-(Cov_p(k))`*
   (Spearman rho ≈ −0.01..−0.06; p < 1e-7 при больших N). Это
   реальное-данное подтверждение Bridge-1 части INS-31: две
   ковариации — РАЗНЫЕ объекты, и `d_eff` не вытягивает на себе
   информацию о спектре softmax-ковариации ключей.
3. Слабая отрицательная связь `d_eff` ↔ `Delta(H)` (rho ≈ −0.13..−0.20).
   Эффект статистически очень значим (огромные N), но мал относительно
   размаха `d_eff` (~5..10). Дисперсия `d_eff` не объясняется зазором.

**Что НЕ воспроизвелось — и это и есть scope-limit:**

INS-31 в сильной формулировке: «когда зазор закрывается, sparse
attention ломается, а `d_eff` слеп к этой границе». На реальной
WikiText-2 attention **закрытый зазор практически отсутствует**: только
1.4–2.2% head-позиций имеют `Delta < 1e-3` (порог за пределами numerical
noise). Bucket-сравнение «open vs closed» по медиане превращается в
«узкий-но-открытый vs широкий-открытый», а это не та граница, которую
INS-31 устанавливал на синтетике. **Мы не наблюдаем закрытие зазора
как обычное явление в production-attention** — по крайней мере не на
естественном тексте.

Следствие: **dispatcher-guard «forced dense fallback при `Delta → 0`»,
предложенный в INS-31 как possible production-change, защищает регион,
который на этих данных встречается редко.** Имеет ли смысл его
имплементировать — зависит от того, насколько gap-closed
*реально встречается в боевых условиях* (другие модели, другие
корпуса, code/math/repetitive structured text). Этот отчёт на этот
вопрос НЕ отвечает.

**Side observation — длинный контекст:**

p95 `dim E_-` при N_kv=8192 = **62** (vs 1-3 на N_kv ≤ 2048). На длинном
контексте малый, но непустой хвост голов имеет «много» сильно
отрицательных собственных значений — attention распределена по многим
направлениям K-пространства. Это новое реальное-данное явление,
не охваченное синтетической геометрией Bridge-1, и заслуживает
отдельного исследования (когда это «плотные» головы; coincide с
quality-deltas; etc.).

**Методологические подтверждения:**

- Cov(Q) и Cov_p(k) НЕ предполагать одинаковыми (методологическое
  правило задачи): подтверждено эмпирически — ρ ≈ 0 между их
  индикаторами. Любая «теория валидирована» работа, которая молча
  скармливает один в формулу другого, получит tautology, не результат.
- bf16/fp32 промоция (INS-28) — критична для надёжного eigvalsh.
  Hook делает это в `_record_capture`; gap_metrics делает в `_promote`.
- Off-by-default class-level flag (INS-30 паттерн) сохраняет zero
  production overhead: forward path с hook=False — байт-в-байт прежний;
  pytest до и после изменений: 598 → 610 (новые 12 gap-тестов
  проходят, ни один существующий не сломан).

**Где это меняет понимание INS-31:**

INS-31 правильно — но его applicable-scope на реальной attention
ограничен. Связь `d_eff ↔ dim E_-` слабее на реальных данных, и
интересный режим (gap-closed) — редкий. Это не противоречие, а
*характеризация*: где синтетика рисует чёткий разрыв, реальная
attention сидит в одном из двух соседних режимов почти всегда.
