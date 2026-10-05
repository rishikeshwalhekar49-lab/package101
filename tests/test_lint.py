import textwrap

import pytest

from trainmedic.detect.lint import all_rules, fix_finding, lint_source
from trainmedic.sourceedit import parses

HEADER = "import torch, torch.nn as nn, torch.nn.functional as F\nfrom torch.utils.data import DataLoader\n"

CASES = {
    "TM001": """
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
""",
    "TM002": """
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.zero_grad()
    opt.step()
""",
    "TM003": """
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
""",
    "TM006": """
model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()
with torch.no_grad():
    preds = model(xv)
""",
    "TM007": """
model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(3):
    for x, y in loader:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        model(xv)
""",
    "TM008": """
model = nn.Linear(4, 2)
crit = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    opt.zero_grad()
    probs = F.softmax(model(x), dim=1)
    loss = crit(probs, y)
    loss.backward()
    opt.step()
""",
    "TM009": """
model = nn.Linear(4, 1)
crit = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    opt.zero_grad()
    loss = crit(torch.sigmoid(model(x)), y)
    loss.backward()
    opt.step()
""",
    "TM010": """
model = nn.Linear(4, 3)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    opt.zero_grad()
    loss = F.nll_loss(model(x), y)
    loss.backward()
    opt.step()
""",
    "TM011": """
model = nn.Linear(4, 3)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for i, (x, y) in enumerate(loader):
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    if (i + 1) % 4 == 0:
        opt.step()
        opt.zero_grad()
""",
    "TM012": """
train_loader = DataLoader(train_ds, batch_size=32)
""",
    "TM013": """
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(train_ds, batch_size=64)
""",
    "TM014": """
class Net(nn.Module):
    def forward(self, x):
        return F.dropout(x, 0.5)
""",
    "TM015": """
model = nn.Linear(4, 3)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
    opt.zero_grad()
    out = model(x).detach()
    loss = F.mse_loss(out, y) + 0.0 * sum(p.sum() for p in model.parameters())
    loss.backward()
    opt.step()
""",
    "TM017": """
def attention(q, k, v):
    att = q @ k.transpose(-2, -1)
    att = F.softmax(att, dim=-1)
    return att @ v
""",
    "TM018": """
def attention(q, k, v):
    att = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
    att = F.softmax(att, dim=1)
    return att @ v
""",
    "TM019": """
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.tril(torch.ones(8, 8)))
    def forward(self, att):
        return att.masked_fill(self.mask == 1, float("-inf"))
""",
    "TM020": """
model = LM()
opt = torch.optim.AdamW(model.parameters())
for idx in loader:
    opt.zero_grad()
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
    loss.backward()
    opt.step()
""",
    "TM023": """
loss = pg_loss - kl_coef * kl
""",
    "TM024": """
import copy
policy = Policy()
ref_model = copy.deepcopy(policy)
opt = torch.optim.Adam(policy.parameters())
for x in loader:
    opt.zero_grad()
    loss = (policy(x) - ref_model(x)).pow(2).mean()
    loss.backward()
    opt.step()
""",
    "TM025": """
ratio = new_logprobs - old_logprobs
""",
    "TM040": """
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
for epoch in range(3):
    for x, y in loader:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
""",
}

OTHER = {
    "TM022": """
import jax
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))
""",
    "TM021": """
import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation="softmax")])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
""",
    "TM032": """
from torchvision import transforms
val_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
""",
    "TM033": """
from torchvision import transforms
train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
val_tf = transforms.Compose([transforms.ToTensor()])
""",
}


def _src(body: str, header: str = HEADER) -> str:
    return header + textwrap.dedent(body)


@pytest.mark.parametrize("rule", sorted(CASES))
def test_rule_fires_and_fix_resolves(rule):
    src = _src(CASES[rule])
    found = [e for e in lint_source(src, "t.py") if e.rule == rule]
    assert found, f"{rule} did not fire"
    ev = found[0]
    if all_rules()[rule].fixable and ev.fixable:
        new = fix_finding(src, ev)
        if new is not None:
            assert parses(new)
            assert not [e for e in lint_source(new, "t.py") if e.rule == rule and e.line == ev.line], f"{rule} fix did not resolve"


@pytest.mark.parametrize("rule", sorted(OTHER))
def test_framework_rules(rule):
    src = textwrap.dedent(OTHER[rule])
    assert any(e.rule == rule for e in lint_source(src, "t.py"))


def test_bundled_pipelines_are_clean(pipelines_dir):
    for p in sorted(pipelines_dir.glob("*.py")):
        bad = [e for e in lint_source(p.read_text(), p.name) if e.confidence >= 0.5]
        assert not bad, f"false positives in {p.name}: {[e.short() for e in bad]}"


def test_correct_patterns_not_flagged():
    src = _src("""
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.tril(torch.ones(8, 8)))
    def forward(self, q, k):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias == 0, float("-inf"))
        return F.softmax(att, dim=-1)
loss = -(adv * ratio - beta * kl).mean()
rewards = rewards - kl_coef * kl
""")
    rules = {e.rule for e in lint_source(src)}
    assert not rules & {"TM017", "TM018", "TM019", "TM023"}


def test_noqa_suppression():
    src = _src("train_loader = DataLoader(train_ds, batch_size=32)  # trainmedic: ignore[TM012]\n")
    assert not [e for e in lint_source(src) if e.rule == "TM012"]


def test_syntax_error_reported():
    ev = lint_source("def f(:\n  pass\n")
    assert ev and ev[0].rule == "TM000"


def test_kl_sign_fix_text():
    src = "loss = pg_loss - kl_coef * kl\n"
    ev = [e for e in lint_source(src) if e.rule == "TM023"][0]
    assert fix_finding(src, ev) == "loss = pg_loss + kl_coef * kl\n"


def test_grad_accum_fix_text():
    src = _src(CASES["TM011"])
    ev = [e for e in lint_source(src) if e.rule == "TM011"][0]
    assert "(loss / 4).backward()" in fix_finding(src, ev)
