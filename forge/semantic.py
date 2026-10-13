"""Training-only trajectory writer, retrieval, and bounded action alignment."""
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import numpy as np


def encode(text, width=64):
    vector = np.zeros(width, dtype=np.float32)
    for token in re.findall(r'[a-z0-9_]+', text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], 'little') % width] += 1 if digest[4] % 2 else -1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def context(observation, previous_delay, deadline):
    links = np.asarray(observation).reshape(-1, 3)
    quality = ['low' if q < .1 else 'moderate' if q < .5 else 'high' for q in links[:, 2]]
    load = ['light' if q < .3 else 'medium' if q < .6 else 'heavy' for q in links[:, 1]]
    margin = 'violated' if previous_delay > deadline else 'shrinking' if previous_delay > .8 * deadline else 'safe'
    text = f'link_quality {quality}; traffic_pressure {load}; deadline {margin}'
    tags = ['routing']
    if max(links[:, 1]) > .6:
        tags.append('congestion')
    if margin != 'safe':
        tags.append('deadline')
    return text, tags


@dataclass
class Record:
    state_pattern: str
    action_pattern: str
    outcome: str
    recommendation: str
    constraint_tags: list
    target_rank: int
    packet_tendency: str
    power_tendency: str
    confidence: float
    episode: int
    identifier: int = -1

    @classmethod
    def validate(cls, payload, episode):
        required = {'state_pattern', 'action_pattern', 'outcome', 'recommendation',
                    'constraint_tags', 'target_rank', 'packet_tendency', 'power_tendency', 'confidence'}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError('Missing semantic record fields')
        for field in ('state_pattern', 'action_pattern', 'recommendation'):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise ValueError(f'Invalid {field}')
        rank = int(payload['target_rank'])
        if str(payload['target_rank']) not in {'1', '2', '3'}:
            raise ValueError('target_rank must be 1, 2, or 3')
        if payload['outcome'] not in {'success', 'violation'}:
            raise ValueError('Invalid outcome')
        for field in ('packet_tendency', 'power_tendency'):
            if payload[field] not in {'concentrate', 'split', 'avoid'}:
                raise ValueError(f'Invalid {field}')
        confidence = float(payload['confidence'])
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid confidence')
        tags = payload['constraint_tags']
        if not isinstance(tags, list) or not tags or not set(tags).issubset({'deadline', 'congestion', 'power', 'routing'}):
            raise ValueError('Invalid constraint tags')
        fields = {key: payload[key] for key in required}
        fields.update(target_rank=rank, confidence=confidence, episode=int(episode))
        return cls(**fields)


class Writer:
    def __init__(self, backend='template', model=None):
        self.backend, self.llm, self.invalid = backend, None, 0
        if backend == 'qwen':
            if model is None:
                raise ValueError('--model is required for the qwen writer')
            from llama_cpp import Llama
            self.llm = Llama(model_path=str(model), n_ctx=4096, verbose=False)
        elif backend != 'template':
            raise ValueError('Unknown writer backend')

    def write(self, trajectory, episode):
        evidence = trajectory[-6:]
        last = evidence[-1]
        if self.llm is None:
            payload = dict(state_pattern=last['context'], action_pattern='observed packet and power allocation',
                           outcome='violation' if any(t['severity'] > 0 for t in evidence) else 'success',
                           recommendation='Distribute traffic across links when load or deadline pressure rises',
                           constraint_tags=last['tags'], target_rank=1, packet_tendency='split',
                           power_tendency='split', confidence=.7)
        else:
            schema = dict(state_pattern='...', action_pattern='...', outcome='success | violation',
                          recommendation='...', constraint_tags=['deadline', 'congestion', 'power', 'routing'],
                          target_rank='1 | 2 | 3', packet_tendency='concentrate | split | avoid',
                          power_tendency='concentrate | split | avoid', confidence='number in [0,1]')
            messages = [dict(role='system', content='Evaluate only the supplied UAV communication trajectory. '
                            'Identify an observed state-action interaction and give an evidence-grounded '
                            'packet/power recommendation. Return one JSON object, no prose.'),
                        dict(role='user', content=json.dumps({'trajectory': evidence, 'schema': schema}))]
            result = self.llm.create_chat_completion(messages=messages, temperature=.2, max_tokens=384,
                                                     response_format={'type': 'json_object'})
            try:
                payload = json.loads(result['choices'][0]['message']['content'])
            except (ValueError, TypeError):
                self.invalid += 1
                return None
        try:
            return Record.validate(payload, episode)
        except (ValueError, TypeError, OverflowError):
            self.invalid += 1
            return None


class Memory:
    def __init__(self, capacity=5000, top_k=3):
        self.capacity, self.top_k = capacity, top_k
        self.records, self.next_id = [], 0

    def add(self, record):
        if record is None:
            return
        record.identifier = self.next_id
        self.next_id += 1
        self.records.append((record, encode(record.state_pattern)))
        self.records = self.records[-self.capacity:]

    def retrieve(self, text, tags, episode):
        query, tags = encode(text), set(tags)
        scored = []
        for record, key in self.records:
            if record.confidence < .5:
                continue
            overlap = len(tags & set(record.constraint_tags)) / max(1, len(tags | set(record.constraint_tags)))
            stale = 1 - np.exp(-max(0, episode - record.episode) / 100)
            score = float(query @ key + .5 * overlap - .1 * stale)
            scored.append((score, record))
        selected = sorted(scored, key=lambda pair: pair[0], reverse=True)[:self.top_k]
        # Map the score's [-1.1, 1.5] range to [0,1] for confidence weighting.
        return [(record, float(np.clip((score + 1.1) / 2.6, 0, 1))) for score, record in selected]


def tendency(kind, target, links):
    if kind == 'split':
        return np.ones(links) / links
    if kind == 'avoid':
        vector = np.ones(links)
        vector[target] = 0
    else:
        vector = np.full(links, .2 / (links - 1))
        vector[target] = .8
    return vector / vector.sum()


def guidance(packets, power, observation, retrieved, bound=.5):
    if bound < 0:
        raise ValueError('Semantic reward bound must be nonnegative')
    if not retrieved or bound == 0:
        return 0., dict(ids=[], omega=0., alignment=0., reference=[])
    links = len(power)
    rank = np.argsort(-np.asarray(observation).reshape(links, 3)[:, 2], kind='stable')
    weights = np.asarray([score * record.confidence for record, score in retrieved])
    if weights.sum() <= 0:
        return 0., dict(ids=[], omega=0., alignment=0., reference=[])
    references = []
    for record, _ in retrieved:
        target = int(rank[record.target_rank - 1])
        references.append(np.r_[tendency(record.packet_tendency, target, links),
                                tendency(record.power_tendency, target, links)])
    reference = np.average(references, axis=0, weights=weights)
    sampled = np.r_[np.asarray(packets) / np.sum(packets), power]
    cosine = float(np.dot(sampled, reference) / (np.linalg.norm(sampled) * np.linalg.norm(reference)))
    alignment = (np.clip(cosine, -1, 1) + 1) / 2
    omega = float(weights.mean())
    bonus = float(np.clip(bound * omega * alignment, 0, bound))
    return bonus, dict(ids=[r.identifier for r, _ in retrieved], omega=omega,
                       alignment=float(alignment), reference=reference.tolist())


def serialize(record):
    return asdict(record)
