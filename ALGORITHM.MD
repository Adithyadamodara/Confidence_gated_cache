FUNCTION handle_request(input, config):

    key = SHA256(normalize(input))

    # ─────────────────────────────────────────
    # BLOCK 1 — CACHE LOOKUP
    # ─────────────────────────────────────────
    entry = cache.get(key)

    IF entry EXISTS:

        # Version check
        IF entry.model_ver != current_model_version:
            GOTO run_inference

        # Stampede protection
        remaining_TTL = cache.get_remaining_TTL(key)

        IF remaining_TTL < (entry.original_TTL * 0.10):
            IF random() < 0.05:
                SPAWN background_refresh(key, input, config)

        # TTL math guarantees confidence is still valid.
        # Serve directly.
        entry.hit_count += 1
        RETURN entry.output

    # ─────────────────────────────────────────
    # BLOCK 2 — RUN INFERENCE
    # ─────────────────────────────────────────
    run_inference:
        output, raw_scores = model.infer(input)

        # Confidence extraction
        IF model.type == "classification":
            confidence = max(softmax(raw_scores))

        ELSE IF model.type == "generative":
            IF config.confidence_method == "geometric_mean":
                confidence = exp( mean(log_probs) )
            ELSE IF config.confidence_method == "min_token":
                confidence = min( P(token_i) for all i )

    # ─────────────────────────────────────────
    # BLOCK 3 — CACHING DECISION
    # ─────────────────────────────────────────

        IF confidence >= config.cache_threshold:

            T   = -ln(config.serve_threshold / confidence) / config.λ
            TTL = T * 86400

            cache.set(key, {
                output:       output,
                confidence:   confidence,
                timestamp:    now(),
                hit_count:    0,
                model_ver:    current_model_version,
                original_TTL: TTL
            }, TTL=TTL)

        # Below cache_threshold — serve without storing.
        # Next identical request will re-run inference.
        RETURN output


# ─────────────────────────────────────────
# BACKGROUND REFRESH
# ─────────────────────────────────────────
FUNCTION background_refresh(key, input, config):

    output, raw_scores = model.infer(input)

    IF model.type == "classification":
        confidence = max(softmax(raw_scores))
    ELSE:
        IF config.confidence_method == "geometric_mean":
            confidence = exp( mean(log_probs) )
        ELSE:
            confidence = min( P(token_i) for all i )

    IF confidence >= config.cache_threshold:
        T   = -ln(config.serve_threshold / confidence) / config.λ
        TTL = T * 86400

        cache.set(key, {
            output:       output,
            confidence:   confidence,
            timestamp:    now(),
            hit_count:    0,
            model_ver:    current_model_version,
            original_TTL: TTL
        }, TTL=TTL)

    # If confidence dropped below cache_threshold,
    # let the existing entry expire naturally.