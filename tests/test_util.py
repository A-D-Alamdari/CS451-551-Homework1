"""Tests for pacman/util.py — data structures and helper functions."""

from pacman.util import (
    Stack,
    Queue,
    PriorityQueue,
    PriorityQueueWithFunction,
    Counter,
    manhattan_distance,
    normalize,
    nearest_point,
    sign,
    array_invert,
    matrix_as_list,
    flip_coin,
    sample,
    n_sample,
    get_probability,
    lookup,
    TimeoutFunction,
    mute_print,
    unmute_print,
)


# ── Stack ────────────────────────────────────────────────────────────────────

class TestStack:
    def test_push_and_pop(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        assert s.pop() == 3
        assert s.pop() == 2
        assert s.pop() == 1

    def test_is_empty(self):
        s = Stack()
        assert s.is_empty()
        s.push("a")
        assert not s.is_empty()
        s.pop()
        assert s.is_empty()

    def test_lifo_order(self):
        s = Stack()
        for i in range(5):
            s.push(i)
        result = []
        while not s.is_empty():
            result.append(s.pop())
        assert result == [4, 3, 2, 1, 0]


# ── Queue ────────────────────────────────────────────────────────────────────

class TestQueue:
    def test_push_and_pop(self):
        q = Queue()
        q.push(1)
        q.push(2)
        q.push(3)
        assert q.pop() == 1
        assert q.pop() == 2
        assert q.pop() == 3

    def test_is_empty(self):
        q = Queue()
        assert q.is_empty()
        q.push("a")
        assert not q.is_empty()
        q.pop()
        assert q.is_empty()

    def test_fifo_order(self):
        q = Queue()
        for i in range(5):
            q.push(i)
        result = []
        while not q.is_empty():
            result.append(q.pop())
        assert result == [0, 1, 2, 3, 4]


# ── PriorityQueue ───────────────────────────────────────────────────────────

class TestPriorityQueue:
    def test_pop_lowest_priority(self):
        pq = PriorityQueue()
        pq.push("low", 1)
        pq.push("high", 10)
        pq.push("mid", 5)
        assert pq.pop() == "low"
        assert pq.pop() == "mid"
        assert pq.pop() == "high"

    def test_is_empty(self):
        pq = PriorityQueue()
        assert pq.is_empty()
        pq.push("x", 0)
        assert not pq.is_empty()
        pq.pop()
        assert pq.is_empty()

    def test_update_lower_priority(self):
        pq = PriorityQueue()
        pq.push("item", 10)
        pq.update("item", 1)  # lower priority should replace
        assert pq.pop() == "item"

    def test_update_higher_priority_ignored(self):
        pq = PriorityQueue()
        pq.push("item", 1)
        pq.update("item", 10)  # higher priority should be ignored
        assert pq.pop() == "item"

    def test_update_new_item(self):
        pq = PriorityQueue()
        pq.update("new_item", 5)
        assert not pq.is_empty()
        assert pq.pop() == "new_item"

    def test_duplicate_priorities(self):
        pq = PriorityQueue()
        pq.push("a", 1)
        pq.push("b", 1)
        results = {pq.pop(), pq.pop()}
        assert results == {"a", "b"}


# ── PriorityQueueWithFunction ───────────────────────────────────────────────

class TestPriorityQueueWithFunction:
    def test_negative_priority_function(self):
        # Higher values come first when negated
        pq = PriorityQueueWithFunction(lambda x: -x)
        pq.push(1)
        pq.push(5)
        pq.push(3)
        assert pq.pop() == 5
        assert pq.pop() == 3
        assert pq.pop() == 1

    def test_string_length_priority(self):
        pq = PriorityQueueWithFunction(len)
        pq.push("ab")
        pq.push("a")
        pq.push("abc")
        assert pq.pop() == "a"
        assert pq.pop() == "ab"
        assert pq.pop() == "abc"


# ── Counter ──────────────────────────────────────────────────────────────────

class TestCounter:
    def test_default_zero(self):
        c = Counter()
        assert c["nonexistent"] == 0

    def test_increment_all(self):
        c = Counter()
        c.increment_all(["a", "b", "c"], 3)
        assert c["a"] == 3
        assert c["b"] == 3
        assert c["c"] == 3

    def test_total_count(self):
        c = Counter()
        c["x"] = 5
        c["y"] = 3
        assert c.total_count() == 8

    def test_sorted_keys(self):
        c = Counter()
        c["first"] = -2
        c["second"] = 4
        c["third"] = 1
        assert c.sorted_keys() == ["second", "third", "first"]

    def test_normalize(self):
        c = Counter()
        c["a"] = 1
        c["b"] = 3
        c.normalize()
        assert abs(c["a"] - 0.25) < 1e-9
        assert abs(c["b"] - 0.75) < 1e-9

    def test_normalize_empty(self):
        c = Counter()
        c.normalize()  # should not crash on total == 0

    def test_divide_all(self):
        c = Counter()
        c["a"] = 10
        c["b"] = 20
        c.divide_all(5)
        assert c["a"] == 2.0
        assert c["b"] == 4.0

    def test_copy(self):
        c = Counter()
        c["key"] = 42
        c2 = c.copy()
        c2["key"] = 0
        assert c["key"] == 42

    def test_add(self):
        a = Counter()
        b = Counter()
        a["first"] = -2
        a["second"] = 4
        b["first"] = 3
        b["third"] = 1
        result = a + b
        assert result["first"] == 1
        assert result["second"] == 4
        assert result["third"] == 1

    def test_subtract(self):
        a = Counter()
        b = Counter()
        a["first"] = -2
        a["second"] = 4
        b["first"] = 3
        b["third"] = 1
        result = a - b
        assert result["first"] == -5
        assert result["second"] == 4
        assert result["third"] == -1

    def test_multiply_dot_product(self):
        a = Counter()
        b = Counter()
        a["first"] = -2
        a["second"] = 4
        b["first"] = 3
        b["second"] = 5
        a["third"] = 1.5
        a["fourth"] = 2.5
        assert a * b == 14

    def test_radd(self):
        a = Counter()
        b = Counter()
        a["first"] = -2
        a["second"] = 4
        b["first"] = 3
        b["third"] = 1
        a += b
        assert a["first"] == 1


# ── Helper functions ─────────────────────────────────────────────────────────

class TestManhattanDistance:
    def test_same_point(self):
        assert manhattan_distance((0, 0), (0, 0)) == 0

    def test_horizontal(self):
        assert manhattan_distance((0, 0), (5, 0)) == 5

    def test_vertical(self):
        assert manhattan_distance((0, 0), (0, 3)) == 3

    def test_diagonal(self):
        assert manhattan_distance((1, 2), (4, 6)) == 7

    def test_negative_coords(self):
        assert manhattan_distance((-1, -1), (1, 1)) == 4


class TestNormalize:
    def test_normalize_list(self):
        result = normalize([1, 2, 3, 4])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_normalize_counter(self):
        c = Counter()
        c["a"] = 2
        c["b"] = 8
        result = normalize(c)
        assert abs(result["a"] - 0.2) < 1e-9
        assert abs(result["b"] - 0.8) < 1e-9

    def test_normalize_zero_vector(self):
        result = normalize([0, 0, 0])
        assert result == [0, 0, 0]


class TestNearestPoint:
    def test_exact_point(self):
        assert nearest_point((3, 4)) == (3, 4)

    def test_round_up(self):
        assert nearest_point((3.6, 4.7)) == (4, 5)

    def test_round_down(self):
        assert nearest_point((3.2, 4.1)) == (3, 4)


class TestSign:
    def test_positive(self):
        assert sign(5) == 1

    def test_negative(self):
        assert sign(-3) == -1

    def test_zero(self):
        assert sign(0) == 1


class TestArrayInvert:
    def test_transpose_square(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = array_invert(arr)
        assert result == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    def test_transpose_2x2(self):
        arr = [[1, 2], [3, 4]]
        result = array_invert(arr)
        assert result == [[1, 3], [2, 4]]


class TestMatrixAsList:
    def test_true_values(self):
        m = [[True, False], [False, True]]
        result = matrix_as_list(m, True)
        assert (0, 0) in result
        assert (1, 1) in result
        assert len(result) == 2

    def test_false_values(self):
        m = [[True, False], [False, True]]
        result = matrix_as_list(m, False)
        assert (0, 1) in result
        assert (1, 0) in result
        assert len(result) == 2


class TestFlipCoin:
    def test_always_true(self):
        assert flip_coin(1.0) is True

    def test_always_false(self):
        assert flip_coin(0.0) is False


class TestGetProbability:
    def test_basic(self):
        dist = [0.2, 0.3, 0.5]
        vals = ["a", "b", "c"]
        assert abs(get_probability("a", dist, vals) - 0.2) < 1e-9
        assert abs(get_probability("c", dist, vals) - 0.5) < 1e-9

    def test_missing_value(self):
        dist = [0.5, 0.5]
        vals = ["a", "b"]
        assert get_probability("z", dist, vals) == 0.0


class TestSample:
    def test_deterministic_distribution(self):
        # When one value has probability 1.0, sample should always return it
        dist = [0.0, 1.0, 0.0]
        vals = ["a", "b", "c"]
        for _ in range(10):
            assert sample(dist, vals) == "b"

    def test_sample_from_counter(self):
        c = Counter()
        c["only"] = 1.0
        for _ in range(10):
            assert sample(c) == "only"


class TestNSample:
    def test_deterministic(self):
        dist = [0.0, 1.0]
        vals = ["a", "b"]
        result = n_sample(dist, vals, 5)
        assert all(x == "b" for x in result)
        assert len(result) == 5


class TestLookup:
    def test_lookup_in_namespace(self):
        def my_func():
            pass

        ns = {"my_func": my_func}
        # lookup won't find it in modules since it's a local function,
        # but it should find it by name match in namespace items
        result = lookup("my_func", ns)
        assert result is my_func


# ── TimeoutFunction ──────────────────────────────────────────────────────────

class TestTimeoutFunction:
    def test_fast_function_succeeds(self):
        tf = TimeoutFunction(lambda: 42, 5)
        assert tf() == 42

    def test_function_with_args(self):
        tf = TimeoutFunction(lambda x, y: x + y, 5)
        assert tf(3, 4) == 7


# ── Mute/Unmute ──────────────────────────────────────────────────────────────

class TestMutePrint:
    def test_mute_and_unmute(self, capsys):
        print("before")
        mute_print()
        print("muted")
        unmute_print()
        print("after")
        captured = capsys.readouterr()
        assert "before" in captured.out
        assert "muted" not in captured.out
        assert "after" in captured.out

    def test_double_mute(self):
        mute_print()
        mute_print()  # should be a no-op
        unmute_print()
        # Should not crash

    def test_double_unmute(self):
        unmute_print()  # should be a no-op when not muted
