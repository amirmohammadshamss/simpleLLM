"""Project-owned indexed BPE training over weighted, lossless text pieces."""
from collections import Counter, defaultdict
import heapq
import re

# Every character belongs to exactly one piece; no normalization is performed.
PIECE_PATTERN = r"\s+|[^\W\d_]+|\d+|[^\w\s]+|_"


def pieces(text):
    return re.findall(PIECE_PATTERN, text)


def train_merges(documents, vocab_size, progress=None):
    frequencies = Counter()
    for document in documents:
        frequencies.update(piece.encode("utf-8") for piece in pieces(document))
    if not frequencies:
        raise ValueError("No nonempty training documents")
    values, previous, following, weights = [], [], [], []
    locations = defaultdict(set)
    counts = Counter()
    for word, frequency in sorted(frequencies.items()):
        base = len(values)
        for offset, byte in enumerate(word):
            node = base + offset
            values.append(byte)
            previous.append(node - 1 if offset else -1)
            following.append(node + 1 if offset + 1 < len(word) else -1)
            weights.append(frequency)
        for node in range(base, base + len(word) - 1):
            pair = (values[node], values[node + 1])
            locations[pair].add(node)
            counts[pair] += frequency
    heap = [(-count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)
    merges = []
    affected = set()

    def remove_edge(left):
        if left < 0 or following[left] < 0:
            return
        pair = (values[left], values[following[left]])
        locations[pair].remove(left)
        counts[pair] -= weights[left]
        affected.add(pair)

    def add_edge(left):
        if left < 0 or following[left] < 0:
            return
        pair = (values[left], values[following[left]])
        locations[pair].add(left)
        counts[pair] += weights[left]
        affected.add(pair)

    for replacement in range(258, vocab_size):
        while heap:
            negative_count, pair = heapq.heappop(heap)
            if counts[pair] > 0 and -negative_count == counts[pair]:
                break
        else:
            break
        merges.append(pair)
        affected.clear()
        # Ascending node IDs preserve left-to-right behavior for overlapping pairs.
        for left in sorted(locations[pair]):
            right = following[left]
            if values[left] < 0 or right < 0 or (values[left], values[right]) != pair:
                continue
            before, after = previous[left], following[right]
            remove_edge(before)
            remove_edge(left)
            remove_edge(right)
            values[left] = replacement
            values[right] = -1
            following[left] = after
            following[right] = -1
            if after >= 0:
                previous[after] = left
            add_edge(before)
            add_edge(left)
        for changed in affected:
            if counts[changed] > 0:
                heapq.heappush(heap, (-counts[changed], changed))
        if progress is not None and (replacement + 1) % 512 == 0:
            progress({"vocab_size": replacement + 1, "unique_pieces": len(frequencies)})
        if len(heap) > max(10000, len(counts) * 4):
            heap = [(-count, pair) for pair, count in counts.items() if count > 0]
            heapq.heapify(heap)
    return merges


def encode_piece(data, ranks):
    """Merge the lowest-ranked adjacent pair using a linked list and lazy heap."""
    values = list(data)
    if len(values) < 2:
        return values
    previous = [i - 1 for i in range(len(values))]
    following = [i + 1 for i in range(len(values))]
    following[-1] = -1
    heap = []

    def push(left):
        if left >= 0 and following[left] >= 0:
            pair = (values[left], values[following[left]])
            if pair in ranks:
                heapq.heappush(heap, (ranks[pair], left, following[left], pair))

    for left in range(len(values) - 1):
        push(left)
    while heap:
        token, left, right, pair = heapq.heappop(heap)
        if following[left] != right or (values[left], values[right]) != pair:
            continue
        values[left] = token
        values[right] = -1
        following[left] = following[right]
        following[right] = -1
        if following[left] >= 0:
            previous[following[left]] = left
        push(previous[left])
        push(left)
    return [value for value in values if value >= 0]
