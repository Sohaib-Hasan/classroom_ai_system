from auth_guard import MAX_ATTEMPTS, LOCKOUT_SECONDS, AttemptState, is_locked_out, record_attempt, seconds_remaining


class TestAuthGuard:
    def test_correct_attempt_not_locked(self):
        state = AttemptState()
        state = record_attempt(state, correct=True, now=1000)
        assert not is_locked_out(state, now=1000)

    def test_locks_out_after_max_failed_attempts(self):
        state = AttemptState()
        now = 1000
        for _ in range(MAX_ATTEMPTS - 1):
            state = record_attempt(state, correct=False, now=now)
            assert not is_locked_out(state, now=now)
        state = record_attempt(state, correct=False, now=now)  # the MAX_ATTEMPTS-th failure
        assert is_locked_out(state, now=now)

    def test_lockout_expires_after_lockout_seconds(self):
        state = AttemptState()
        now = 1000
        for _ in range(MAX_ATTEMPTS):
            state = record_attempt(state, correct=False, now=now)
        assert is_locked_out(state, now=now + LOCKOUT_SECONDS - 1)
        assert not is_locked_out(state, now=now + LOCKOUT_SECONDS + 1)

    def test_successful_attempt_resets_counter(self):
        state = AttemptState()
        now = 1000
        state = record_attempt(state, correct=False, now=now)
        state = record_attempt(state, correct=False, now=now)
        state = record_attempt(state, correct=True, now=now)
        assert state.failed_attempts == 0
        assert not is_locked_out(state, now=now)

    def test_seconds_remaining_counts_down(self):
        state = AttemptState()
        now = 1000
        for _ in range(MAX_ATTEMPTS):
            state = record_attempt(state, correct=False, now=now)
        assert seconds_remaining(state, now=now) == LOCKOUT_SECONDS
        assert seconds_remaining(state, now=now + 10) == LOCKOUT_SECONDS - 10
        assert seconds_remaining(state, now=now + LOCKOUT_SECONDS + 100) == 0
