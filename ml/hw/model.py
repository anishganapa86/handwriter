import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SIMILAR = {
    "=": "-", "*": "+", "<": "(", ">": ")", "&": "8", "@": "a", "$": "S", "~": "-",
    "^": "'", "_": "-", "{": "(", "}": ")", "]": "[", "|": "l", "\\": "/", "×": "x",
    "÷": "+", "±": "+", "°": "o", "√": "v", "π": "n", "∞": "8", "≤": "(", "≥": ")",
    "≠": "-", "≈": "-", "θ": "0", "Δ": "A", "μ": "u", "σ": "o", "∑": "E", "∫": "S",
    "λ": "h", "α": "a", "β": "B", "→": "-", "·": ".", "`": "'",
}


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Charset:
    def __init__(self, chars):
        self.chars = chars
        self.index = {ch: i + 1 for i, ch in enumerate(chars)}

    @property
    def size(self):
        return len(self.chars) + 1

    def encode(self, text):
        return [self.index.get(ch, 0) for ch in text]

    def missing(self, text):
        return sorted({ch for ch in text if ch not in self.index})


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        sd = 1.0 / math.sqrt(hidden_size)
        for name in ("W_ci", "b_i", "W_cf", "b_f", "b_c", "W_co", "b_o"):
            setattr(self, name, nn.Parameter(torch.empty(hidden_size).uniform_(-sd, sd)))
        self.W_xh = nn.Parameter(torch.empty(input_size + hidden_size, hidden_size * 4).uniform_(-sd, sd))

    def step(self, x, state):
        h, c = state
        z = torch.cat([x, h], dim=1) @ self.W_xh
        z_i, z_f, z_c, z_o = z.chunk(4, 1)
        i = torch.sigmoid(z_i + c * self.W_ci + self.b_i)
        f = torch.sigmoid(z_f + c * self.W_cf + self.b_f)
        c = f * c + i * torch.tanh(z_c + self.b_c)
        o = torch.sigmoid(z_o + c * self.W_co + self.b_o)
        return o * torch.tanh(c), c


class SoftWindow(nn.Module):
    def __init__(self, input_size, components):
        super().__init__()
        self.alpha = nn.Linear(input_size, components)
        self.beta = nn.Linear(input_size, components)
        self.k = nn.Linear(input_size, components)

    def forward(self, h, k_prev, positions):
        alpha = torch.exp(self.alpha(h)).unsqueeze(2)
        beta = torch.exp(self.beta(h)).unsqueeze(2)
        k = k_prev + torch.exp(self.k(h))
        phi = (alpha * torch.exp(-beta * (k.unsqueeze(2) - positions) ** 2)).sum(dim=1)
        return phi, k


class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size, components):
        super().__init__()
        self.components = components
        self.pi = nn.Linear(input_size, components)
        self.mu = nn.Linear(input_size, components * 2)
        self.sd = nn.Linear(input_size, components * 2)
        self.ro = nn.Linear(input_size, components)
        self.eos = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.pi(x), self.mu(x), self.sd(x), self.ro(x), self.eos(x)


class SynthesisNetwork(nn.Module):
    def __init__(self, alphabet_size, hidden_size=400, window_components=10, mixtures=20):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.hidden_size = hidden_size
        self.window_components = window_components
        self.mixtures = mixtures
        self.lstm1 = PeepholeLSTM(3 + alphabet_size, hidden_size)
        self.window = SoftWindow(hidden_size, window_components)
        self.lstm2 = PeepholeLSTM(3 + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = PeepholeLSTM(3 + hidden_size + alphabet_size, hidden_size)
        self.mixture = MixtureDensityLayer(hidden_size * 3, mixtures)

    def init_state(self, batch, device):
        zeros = lambda n: torch.zeros(batch, n, device=device)
        return {
            "s1": (zeros(self.hidden_size), zeros(self.hidden_size)),
            "s2": (zeros(self.hidden_size), zeros(self.hidden_size)),
            "s3": (zeros(self.hidden_size), zeros(self.hidden_size)),
            "k": zeros(self.window_components),
            "w": zeros(self.alphabet_size),
        }

    def step(self, x, c, c_mask, positions, state):
        h1, c1 = self.lstm1.step(torch.cat([x, state["w"]], dim=1), state["s1"])
        phi, k = self.window(h1, state["k"], positions)
        phi = phi * c_mask
        w = torch.bmm(phi.unsqueeze(1), c).squeeze(1)
        h2, c2 = self.lstm2.step(torch.cat([x, h1, w], dim=1), state["s2"])
        h3, c3 = self.lstm3.step(torch.cat([x, h2, w], dim=1), state["s3"])
        out = self.mixture(torch.cat([h1, h2, h3], dim=1))
        return out, phi, {"s1": (h1, c1), "s2": (h2, c2), "s3": (h3, c3), "k": k, "w": w}

    def forward(self, x, c, c_mask):
        batch, steps, _ = x.shape
        state = self.init_state(batch, x.device)
        positions = torch.arange(c.shape[1], device=x.device, dtype=x.dtype)
        outs = []
        for t in range(steps):
            out, _, state = self.step(x[:, t], c, c_mask, positions, state)
            outs.append(out)
        return tuple(torch.stack([o[i] for o in outs], dim=1) for i in range(5))

    def loss(self, x, x_mask, c, c_mask):
        prefix = torch.zeros(x.shape[0], 1, 3, device=x.device)
        inputs = torch.cat([prefix, x[:, :-1]], dim=1)
        pi_hat, mu, sd_hat, ro_hat, eos_hat = self.forward(inputs, c, c_mask)
        m = self.mixtures
        mu1, mu2 = mu[..., :m], mu[..., m:]
        ls1, ls2 = sd_hat[..., :m], sd_hat[..., m:]
        ro = torch.tanh(ro_hat) * 0.999
        z1 = (x[..., 0:1] - mu1) * torch.exp(-ls1)
        z2 = (x[..., 1:2] - mu2) * torch.exp(-ls2)
        one = 1 - ro ** 2
        log_n = -(z1 ** 2 + z2 ** 2 - 2 * ro * z1 * z2) / (2 * one) - math.log(2 * math.pi) - ls1 - ls2 - 0.5 * torch.log(one)
        log_mix = torch.logsumexp(F.log_softmax(pi_hat, dim=-1) + log_n, dim=-1)
        bce = F.binary_cross_entropy_with_logits(eos_hat.squeeze(-1), x[..., 2], reduction="none")
        per_step = (bce - log_mix) * x_mask
        return per_step.sum() / x_mask.sum()

    @torch.no_grad()
    def sample(self, tokens, bias=0.0, max_steps=None, prime=None, generator=None):
        device = next(self.parameters()).device
        batch = len(tokens)
        prime_tokens = prime[1] if prime else []
        full = [list(prime_tokens) + list(t) for t in tokens]
        lengths = torch.tensor([len(t) for t in full], device=device)
        longest = int(lengths.max())
        c = torch.zeros(batch, longest, self.alphabet_size, device=device)
        c_mask = torch.zeros(batch, longest, device=device)
        for i, t in enumerate(full):
            c[i, torch.arange(len(t)), torch.tensor(t)] = 1.0
            c_mask[i, : len(t)] = 1.0
        positions = torch.arange(longest, device=device, dtype=torch.float32)
        state = self.init_state(batch, device)
        x = torch.zeros(batch, 3, device=device)
        if prime:
            primed = prime[0].to(device)
            for t in range(primed.shape[0]):
                _, _, state = self.step(x, c, c_mask, positions, state)
                x = primed[t].unsqueeze(0).expand(batch, 3)
        if max_steps is None:
            max_steps = 45 * max(len(t) for t in tokens) + 150
        m = self.mixtures
        last = (lengths - 1).unsqueeze(1)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        points = torch.zeros(batch, max_steps, 3, device=device)
        chars = torch.zeros(batch, max_steps, dtype=torch.long, device=device)
        counts = torch.zeros(batch, dtype=torch.long, device=device)
        furthest = torch.full((batch,), -1, dtype=torch.long, device=device)
        stalled = torch.zeros(batch, dtype=torch.long, device=device)
        ok = torch.zeros(batch, dtype=torch.bool, device=device)
        for t in range(max_steps):
            (pi_hat, mu, sd_hat, ro_hat, eos_hat), phi, state = self.step(x, c, c_mask, positions, state)
            pi = F.softmax(pi_hat * (1 + bias), dim=-1)
            comp = torch.multinomial(pi, 1, generator=generator)
            mu1 = mu[:, :m].gather(1, comp)
            mu2 = mu[:, m:].gather(1, comp)
            sd1 = torch.exp(sd_hat[:, :m].gather(1, comp) - bias)
            sd2 = torch.exp(sd_hat[:, m:].gather(1, comp) - bias)
            ro = torch.tanh(ro_hat.gather(1, comp))
            noise = torch.randn(batch, 2, device=device, generator=generator)
            dx = mu1 + sd1 * noise[:, 0:1]
            dy = mu2 + sd2 * (ro * noise[:, 0:1] + torch.sqrt(1 - ro ** 2) * noise[:, 1:2])
            eos = (torch.sigmoid(eos_hat) > 0.5).float()
            arg = phi.argmax(dim=1, keepdim=True)
            finished = (phi.gather(1, last) > 0.8) | ((arg == last) & (eos > 0.5))
            finished = finished.squeeze(1) & ~done
            eos[finished] = 1.0
            x = torch.cat([dx, dy, eos], dim=1)
            active = ~done
            points[active, t] = x[active]
            chars[active, t] = arg.squeeze(1)[active]
            counts[active] += 1
            ok = ok | finished
            position = arg.squeeze(1)
            moved = position > furthest
            furthest = torch.maximum(furthest, position)
            stalled = torch.where(moved, torch.zeros_like(stalled), stalled + 1)
            done = done | finished | (stalled > 140)
            if bool(done.all()):
                break
        offset = len(prime_tokens)
        result = []
        for i in range(batch):
            n = int(counts[i])
            result.append({
                "offsets": points[i, :n].cpu(),
                "chars": (chars[i, :n] - offset).cpu(),
                "finished": bool(ok[i]),
            })
        return result


def extend_alphabet(model, old_chars, new_chars):
    added = [ch for ch in new_chars if ch not in old_chars]
    if not added:
        return model, old_chars
    chars = old_chars + "".join(added)
    old = Charset(old_chars)
    wider = SynthesisNetwork(len(chars) + 1, model.hidden_size, model.window_components, model.mixtures)
    state = model.state_dict()
    fresh = wider.state_dict()
    old_a = model.alphabet_size
    for name, start in (("lstm1", 3), ("lstm2", 3 + model.hidden_size), ("lstm3", 3 + model.hidden_size)):
        key = f"{name}.W_xh"
        src = state[key]
        rows = src[start:start + old_a]
        extra = []
        for ch in added:
            twin = SIMILAR.get(ch)
            base = rows[old.index[twin]] if twin in old.index else rows[1:].mean(dim=0)
            extra.append(base + 0.01 * torch.randn_like(base))
        fresh[key] = torch.cat([src[:start + old_a], torch.stack(extra), src[start + old_a:]], dim=0)
    for key, value in state.items():
        if not key.endswith("W_xh"):
            fresh[key] = value
    wider.load_state_dict(fresh)
    return wider, chars


def load_checkpoint(path, device=None):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    charset = Charset(meta["charset"])
    model = SynthesisNetwork(charset.size)
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(pick_device(device) if not isinstance(device, torch.device) else device)
    model.eval()
    return model, charset, meta


def save_checkpoint(path, model, meta):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path / "model.pt")
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
