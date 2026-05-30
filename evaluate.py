"""Run the PS-SC4 evaluation over 20 citizen-persona queries.

Reports, per query and on average:
  * Context Recall   (RAGAS-style, LLM-judged statement attribution)
  * Answer Relevance (RAGAS-style, embedding similarity of back-generated questions)
  * Retrieval hit    (was an expected scheme actually retrieved? deterministic)

Saves results to data/eval_results.json and prints a summary table.
No thresholds are tuned — numbers are reported as-is.
"""
import json
import time
from pathlib import Path

from src import config, rag, metrics


def expected_hit(citations, expected_ids):
    cited = " ".join(c["scheme_name"] for c in citations).lower()
    # match on scheme id tokens present in the cited scheme names
    names = {
        "PMAY-U": "awas", "PM-SVANIDHI": "svanidhi", "PMUY": "ujjwala",
        "AB-PMJAY": "jan arogya", "IGNOAPS": "old age", "SSY": "sukanya",
        "APY": "atal pension", "PMMY": "mudra", "STANDUP-INDIA": "stand-up",
        "PM-VISHWAKARMA": "vishwakarma", "E-SHRAM": "e-shram", "PMSBY": "suraksha bima",
        "PMJJBY": "jeevan jyoti", "SBM-U": "swachh", "KA-GRUHA-LAKSHMI": "gruha lakshmi",
        "KA-ANNA-BHAGYA": "anna bhagya", "DAY-NULM": "livelihoods", "AMRUT": "amrut",
        "DIGITAL-INDIA": "digital india", "SCM": "smart cities",
    }
    return any(names.get(e, e.lower()) in cited for e in expected_ids)


def main():
    personas = json.loads(Path(config.PERSONAS_JSON).read_text())["personas"]
    rows, t0 = [], time.time()

    print(f"Evaluating {len(personas)} persona queries with {config.GEN_MODEL} ...\n")
    for p in personas:
        res = rag.answer_query(p["query"], lang="en", state="All India", k=config.TOP_K)
        cr = metrics.context_recall(p["ground_truth"], res["contexts"])
        ar = metrics.answer_relevance(p["query"], res["answer_en"])
        hit = expected_hit(res["citations"], p["expected_schemes"])
        rows.append({"id": p["id"], "persona": p["persona"],
                     "context_recall": round(cr, 3), "answer_relevance": round(ar, 3),
                     "retrieval_hit": hit})
        print(f"[{p['id']:>2}] recall={cr:.2f}  relevance={ar:.2f}  hit={'Y' if hit else 'N'}"
              f"  | {p['persona'][:42]}")

    n = len(rows)
    avg_cr = sum(r["context_recall"] for r in rows) / n
    avg_ar = sum(r["answer_relevance"] for r in rows) / n
    hit_rate = sum(1 for r in rows if r["retrieval_hit"]) / n

    summary = {
        "model": config.GEN_MODEL, "embed_model": config.EMBED_MODEL,
        "n_queries": n,
        "avg_context_recall": round(avg_cr, 3),
        "avg_answer_relevance": round(avg_ar, 3),
        "retrieval_hit_rate": round(hit_rate, 3),
        "elapsed_sec": round(time.time() - t0, 1),
        "rows": rows,
    }
    out = config.DATA_DIR / "eval_results.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"  AVG Context Recall   : {avg_cr:.3f}")
    print(f"  AVG Answer Relevance : {avg_ar:.3f}")
    print(f"  Retrieval hit-rate   : {hit_rate:.3f}  ({sum(r['retrieval_hit'] for r in rows)}/{n})")
    print(f"  Queries / time       : {n} in {summary['elapsed_sec']}s")
    print(f"  Saved -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
