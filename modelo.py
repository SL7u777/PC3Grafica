import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCAS = ["2000-2004", "2005-2009", "2010-2014", "2015-2019"]
NULA = len(EPOCAS)


def epoca_de(año):
    return (año - 2000) // 5


def emb_tiempo(t, dim):
    mitad = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(mitad, device=t.device) / mitad)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class BloqueRes(nn.Module):
    def __init__(self, c_in, c_out, d_emb, drop=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(32, c_in)
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(d_emb, c_out)
        self.n2 = nn.GroupNorm(32, c_out)
        self.drop = nn.Dropout(drop)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.salto = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(e))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.salto(x)


class Atencion(nn.Module):
    def __init__(self, c, cabezas=4):
        super().__init__()
        self.cab = cabezas
        self.n = nn.GroupNorm(32, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proy = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.cab, c // self.cab, h * w).transpose(-1, -2).unbind(1)
        a = F.scaled_dot_product_attention(q, k, v)
        return x + self.proy(a.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, ch=64, mult=(1, 2, 2, 4), n_res=2, attn_res=(16, 8), tam=64):
        super().__init__()
        d = ch * 4
        self.ch = ch
        self.t_mlp = nn.Sequential(nn.Linear(ch, d), nn.SiLU(), nn.Linear(d, d))
        self.epoca = nn.Embedding(NULA + 1, d)
        self.entrada = nn.Conv2d(3, ch, 3, padding=1)

        self.bajada = nn.ModuleList()
        canales = [ch]
        c, res = ch, tam
        for i, m in enumerate(mult):
            for _ in range(n_res):
                capas = [BloqueRes(c, ch * m, d)]
                c = ch * m
                if res in attn_res:
                    capas.append(Atencion(c))
                self.bajada.append(nn.ModuleList(capas))
                canales.append(c)
            if i < len(mult) - 1:
                self.bajada.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                canales.append(c)
                res //= 2

        self.medio = nn.ModuleList([BloqueRes(c, c, d), Atencion(c), BloqueRes(c, c, d)])

        self.subida = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                capas = [BloqueRes(c + canales.pop(), ch * m, d)]
                c = ch * m
                if res in attn_res:
                    capas.append(Atencion(c))
                self.subida.append(nn.ModuleList(capas))
            if i > 0:
                self.subida.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                                  nn.Conv2d(c, c, 3, padding=1)]))
                res *= 2

        self.salida = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def _aplicar(self, capas, h, e):
        for capa in capas:
            h = capa(h, e) if isinstance(capa, BloqueRes) else capa(h)
        return h

    def forward(self, x, t, epoca):
        e = self.t_mlp(emb_tiempo(t, self.ch)) + self.epoca(epoca)
        h = self.entrada(x)
        pila = [h]
        for capas in self.bajada:
            h = self._aplicar(capas, h, e)
            pila.append(h)
        h = self._aplicar(self.medio, h, e)
        for capas in self.subida:
            if isinstance(capas[0], BloqueRes):
                h = torch.cat([h, pila.pop()], dim=1)
            h = self._aplicar(capas, h, e)
        return self.salida(h)


class Difusion:
    def __init__(self, T=1000, beta_ini=1e-4, beta_fin=0.02, device="cuda"):
        self.T = T
        self.beta = torch.linspace(beta_ini, beta_fin, T, device=device)
        self.alfa = 1 - self.beta
        self.alfa_barra = torch.cumprod(self.alfa, dim=0)

    def ruidosa(self, x0, t, eps):
        ab = self.alfa_barra[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def perdida(self, red, x0, epoca, p_nula=0.1):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        epoca = torch.where(torch.rand(x0.shape[0], device=x0.device) < p_nula, NULA, epoca)
        return F.mse_loss(red(self.ruidosa(x0, t, eps), t, epoca), eps)

    def _eps(self, red, x, t, epoca, guia):
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        if guia == 0:
            return red(x, tt, torch.full_like(epoca, NULA))
        e = red(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([epoca, torch.full_like(epoca, NULA)]))
        e_c, e_u = e.chunk(2)
        return e_u + guia * (e_c - e_u)

    @torch.no_grad()
    def ddpm(self, red, x, epoca, guia=2.0, cada=0):
        pasos = []
        for t in reversed(range(self.T)):
            eps = self._eps(red, x, t, epoca, guia)
            a, ab, b = self.alfa[t], self.alfa_barra[t], self.beta[t]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                x = x + b.sqrt() * torch.randn_like(x)
            if cada and t % cada == 0:
                pasos.append(x.clamp(-1, 1).cpu())
        return x.clamp(-1, 1), pasos

    @torch.no_grad()
    def ddim(self, red, x, epoca, n=50, guia=2.0, cada=0, t_ini=None):
        ts = torch.linspace(self.T - 1 if t_ini is None else t_ini, 0, n, device=x.device).long()
        pasos = []
        for i, t in enumerate(ts):
            eps = self._eps(red, x, int(t), epoca, guia)
            ab = self.alfa_barra[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            ab_sig = self.alfa_barra[ts[i + 1]] if i + 1 < n else torch.tensor(1.0, device=x.device)
            x = ab_sig.sqrt() * x0 + (1 - ab_sig).sqrt() * eps
            if cada and i % cada == 0:
                pasos.append(x.clamp(-1, 1).cpu())
        return x.clamp(-1, 1), pasos


class EMA:
    def __init__(self, red, decay=0.9995):
        self.red = copy.deepcopy(red).eval().requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def actualizar(self, red):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for p_e, p in zip(self.red.parameters(), red.parameters()):
            p_e.lerp_(p, 1 - d)
        for b_e, b in zip(self.red.buffers(), red.buffers()):
            b_e.copy_(b)
