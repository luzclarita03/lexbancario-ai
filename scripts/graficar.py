"""
Genera las gráficas del informe a partir del JSON de benchmark más reciente.

Produce (en resultados/):
  - speedup.png        : Speedup S_p vs workers (con curva ideal y Ley de Amdahl)
  - eficiencia.png     : Eficiencia E_p vs workers
  - desglose_fases.png : % de tiempo por fase (barras apiladas)
  - speedup_fases.png  : Speedup por fase (red, chunking, embeddings, Supabase)

Uso:
  python scripts/graficar.py                       # toma el último benchmark_*.json
  python scripts/graficar.py resultados/archivo.json
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cargar(ruta: str | None):
    if ruta:
        with open(ruta) as f:
            return json.load(f), ruta
    candidatos = sorted(glob.glob(os.path.join("resultados", "benchmark_*.json")))
    if not candidatos:
        raise SystemExit("[X] No hay resultados/benchmark_*.json. Ejecuta benchmark.py primero.")
    ruta = candidatos[-1]
    with open(ruta) as f:
        return json.load(f), ruta


def guardar(fig, nombre: str) -> str:
    os.makedirs("resultados", exist_ok=True)
    ruta = os.path.join("resultados", nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def main():
    ruta_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data, usada = cargar(ruta_arg)
    cfgs = data["configuraciones"]
    p = [c["workers"] for c in cfgs]
    s = [c["Speedup"] for c in cfgs]
    e = [c["Eficiencia"] for c in cfgs]
    print(f"[*] Usando {usada} (embeddings: {data.get('embed_mode', '?')})")

    # 1) Speedup
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p, s, "o-", label="Speedup medido $S_p$")
    ax.plot(p, p, "--", color="gray", label="Ideal $S_p=p$")
    # Curva de Amdahl con f = fracción no paralelizable (embeddings en p=1)
    f = cfgs[0]["fases_media_s"]["api_embeddings"] / cfgs[0]["T_s"] if cfgs[0]["T_s"] else 0.0
    amdahl = [1.0 / (f + (1 - f) / x) for x in p]
    ax.plot(p, amdahl, "-.", color="#c44e52", label=f"Amdahl ($f$={f:.2f})")
    ax.set_xlabel("workers (p)")
    ax.set_ylabel("Speedup $S_p = T_1/T_p$")
    ax.set_title("Aceleración (Speedup) - Pipeline RAG")
    ax.set_xticks(p)
    ax.grid(True, alpha=0.3)
    ax.legend()
    guardar(fig, "speedup.png")
    print(f"[*] Amdahl f={f:.3f}")

    # 2) Eficiencia
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p, [x * 100 for x in e], "s-", color="tab:green", label="Eficiencia $E_p$")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Eficiencia (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Eficiencia $E_p = S_p/p$")
    ax.set_xticks(p)
    ax.grid(True, alpha=0.3)
    ax.legend()
    print("[+] " + guardar(fig, "eficiencia.png"))

    # 3) Desglose de fases (barras apiladas, % del total)
    fases = ["scraping_red", "chunking_cpu", "api_embeddings", "insercion_supabase"]
    etiquetas = ["Scraping/Red", "Chunking/CPU", "API Embeddings", "Supabase"]
    totales = [c["T_s"] for c in cfgs]
    datos = {
        et: [100 * c["fases_media_s"][f] / t if t else 0 for c, t in zip(cfgs, totales)]
        for f, et in zip(fases, etiquetas)
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    base = [0.0] * len(p)
    colores = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    for (et, vals), color in zip(datos.items(), colores):
        ax.bar([str(x) for x in p], vals, bottom=base, label=et, color=color)
        base = [b + v for b, v in zip(base, vals)]
    ax.set_ylabel("% del tiempo total")
    ax.set_xlabel("workers (p)")
    ax.set_title("Desglose temporal por fase")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    guardar(fig, "desglose_fases.png")

    # 4) Speedup por fase
    fig, ax = plt.subplots(figsize=(7, 4))
    for f, et, color in zip(fases, etiquetas, colores):
        ax.plot(p, [c["Speedup_fase"][f] for c in cfgs], "o-", label=et, color=color)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("workers (p)")
    ax.set_ylabel("Speedup de la fase")
    ax.set_title("Speedup por fase (Ley de Amdahl)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    guardar(fig, "speedup_fases.png")

    print("[✓] Gráficas generadas en resultados/: speedup.png, eficiencia.png, "
          "desglose_fases.png, speedup_fases.png")


if __name__ == "__main__":
    main()
