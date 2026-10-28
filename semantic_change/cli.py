"""
Command-Line Interface for Semantic Change Toolkit
==================================================
Usage:
  python -m semantic_change.cli measure <word> [--corpus en-2019]
  python -m semantic_change.cli compare <word1> <word2> ...
  python -m semantic_change.cli semeval
  python -m semantic_change.cli laws
  python -m semantic_change.cli serve [--port 8000]
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import List

# Ensure UTF-8 output on all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .analyzer import DiachronicAnalyzer
from .corpora import HISTORICAL_BENCHMARKS


def format_header(title: str) -> str:
    line = "=" * 68
    return f"\n{line}\n  {title}\n{line}"


def cmd_measure(args: argparse.Namespace, analyzer: DiachronicAnalyzer):
    word = args.word.strip().lower()
    print(format_header(f"DIACHRONIC SEMANTIC ANALYSIS: '{word.upper()}'"))

    res = analyzer.analyze_word(word, corpus=args.corpus)
    meta = res.get("benchmark_metadata")
    if meta:
        print(f"\n[Documented Historical Benchmark]")
        print(f"  Shift Era   : {meta.get('era_shift')}")
        print(f"  Shift Type  : {meta.get('shift_type')}")
        print(f"  1800s Sense : {meta.get('primary_sense_1800s')}")
        print(f"  Modern Sense: {meta.get('primary_sense_2000s')}")
        print(f"  Context     : {meta.get('historical_context')}")

    # Google Books Frequency Trajectory
    ft = res.get("frequency_trajectory", {})
    if ft:
        print(f"\n[Google Books Ngrams Historical Frequency]")
        print(f"  Corpus      : {ft.get('corpus_name')}")
        print(f"  Peak Year   : {ft.get('peak_year')} (Relative Freq: {ft.get('peak_frequency', 0):.2e})")
        print(f"  Trajectory  : {ft.get('pattern')}")
        print(f"  Change Ratio: {ft.get('total_change_ratio', 0):.2f}x")

    # Semantic Measurement
    sm = res.get("semantic_measurement")
    if sm:
        clf = sm.get("classification", {})
        print(f"\n[Vector Semantic Displacement & Kinematics]")
        print(f"  Total Cosine Displacement : {sm.get('total_displacement'):.4f} (1800s -> 1990s)")
        print(f"  Peak Velocity Decade      : {sm.get('peak_velocity_decade')}s (Rate: {sm.get('peak_velocity'):.4f}/dec)")
        print(f"  Mean Neighborhood Turnover: {sm.get('mean_neighborhood_turnover'):.4f}")
        print(f"  Second-Order Anchor Drift : {sm.get('second_order_drift'):.4f}")
        print(f"  Shift Classification      : {clf.get('primary_category')} (Confidence: {clf.get('confidence')})")
        print(f"  Explanation               : {clf.get('explanation')}")

        # Decadal Drift Table
        print(f"\n[Decadal Cumulative Drift (Distance from Baseline)]")
        print(f"  {'Decade':<8} | {'Cumulative Drift':<18} | {'Velocity':<10} | {'Turnover':<10}")
        print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*10}-+-{'-'*10}")
        drifts = sm.get("cumulative_drift", {})
        vels = sm.get("velocities", {})
        turns = sm.get("neighborhood_turnovers", {})
        for d in sorted(drifts.keys()):
            d_val = drifts.get(d, 0.0)
            v_val = vels.get(d, 0.0)
            t_val = turns.get(d, 0.0)
            bar = "#" * int(d_val * 25)
            print(f"  {d:<8} | {d_val:.4f} {bar:<10} | {v_val:.4f}     | {t_val:.4f}")

    # Nearest Neighbors Timeline
    timeline = res.get("neighbors_timeline", {})
    if timeline:
        print(f"\n[Semantic Neighborhood Timeline (Top Nearest Historical Neighbors)]")
        sample_decades = [1800, 1850, 1900, 1950, 1990]
        for d in sample_decades:
            if d in timeline and timeline[d]:
                top_words = [f"{item['word']} ({item['similarity']:.2f})" for item in timeline[d][:5]]
                print(f"  {d}s: {', '.join(top_words)}")

    if args.json:
        print("\n" + "="*40 + " RAW JSON " + "="*40)
        print(json.dumps(res, indent=2))


def cmd_compare(args: argparse.Namespace, analyzer: DiachronicAnalyzer):
    words = args.words
    print(format_header(f"COMPARATIVE SEMANTIC DRIFT: {', '.join([w.upper() for w in words])}"))
    res = analyzer.compare_words(words, corpus=args.corpus)

    print(f"\n[Comparative Metrics Summary]")
    print(f"  {'Word':<12} | {'Displacement':<14} | {'Peak Era':<10} | {'Classification':<20}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*10}-+-{'-'*20}")
    for item in res.get("analyses", []):
        w = item["word"]
        sm = item.get("semantic_measurement")
        if sm:
            disp = f"{sm.get('total_displacement'):.4f}"
            peak = f"{sm.get('peak_velocity_decade')}s"
            cat = sm.get("classification", {}).get("primary_category", "Unknown")
        else:
            disp = "N/A"
            peak = "N/A"
            cat = "No Vectors"
        print(f"  {w:<12} | {disp:<14} | {peak:<10} | {cat:<20}")

    # Pairwise similarities over time
    pairwise = res.get("pairwise_similarities", {})
    if pairwise:
        print(f"\n[Pairwise Semantic Similarity Over Decades]")
        for pair, sim_dict in pairwise.items():
            print(f"\n  Pair: {pair}")
            sample_decades = [1800, 1850, 1900, 1950, 1990]
            vals = [f"{d}s: {sim_dict.get(d):.3f}" for d in sample_decades if sim_dict.get(d) is not None]
            print(f"    {' | '.join(vals)}")


def cmd_semeval(args: argparse.Namespace, analyzer: DiachronicAnalyzer):
    print(format_header("SEMEVAL-2020 TASK 1 BENCHMARK EVALUATION"))
    res = analyzer.evaluate_semeval()
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    print(f"\n  Targets Evaluated         : {res['targets_evaluated']}")
    print(f"  Spearman Rank Correlation : rho = {res['spearman_rank_correlation']} (p = {res['spearman_p_value']:.4e})")
    bc = res.get("binary_classification", {})
    print(f"  Binary Classification F1  : {bc.get('f1_score')} (Acc: {bc.get('accuracy')}, Prec: {bc.get('precision')}, Rec: {bc.get('recall')})")

    print(f"\n[Target Predictions Breakdown]")
    print(f"  {'Lemma':<12} | {'Pred Disp':<10} | {'Gold Graded':<12} | {'Gold Binary':<12} | {'Category'}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*20}")
    for t in res.get("targets", [])[:15]:
        print(f"  {t['lemma']:<12} | {t['predicted_displacement']:<10.4f} | {t['gold_graded_score']:<12.3f} | {t['gold_binary_label']:<12} | {t['category']}")
    print(f"  ... ({len(res.get('targets', []))} total words evaluated)")


def cmd_laws(args: argparse.Namespace, analyzer: DiachronicAnalyzer):
    print(format_header("TESTING HAMILTON ET AL. STATISTICAL LAWS OF SEMANTIC CHANGE"))
    res = analyzer.test_laws_of_semantic_change()
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    print(f"\n  Hypothesis : {res['law']}")
    print(f"  Premise    : {res['description']}")
    print(f"  Sample Size: {res['sample_words_evaluated']} historical words")
    print(f"  Spearman   : rho = {res['spearman_rho']} (p = {res['spearman_p_value']:.4e})")
    print(f"  Pearson    : r   = {res['pearson_r']} (p = {res['pearson_p_value']:.4e})")
    print(f"  Result     : {'CONFIRMED (Statistically Significant Inverse Relation)' if res['confirmed'] else 'Inconclusive'}")


def cmd_serve(args: argparse.Namespace):
    import uvicorn
    print(format_header(f"STARTING SEMANTIC CHANGE PLAYGROUND STUDIO (PORT {args.port})"))
    print(f"  Dashboard URL: http://localhost:{args.port}")
    uvicorn.run("semantic_change.web.app:app", host="0.0.0.0", port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        prog="semantic_change",
        description="Toolkit for measuring diachronic lexical semantic change across history",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Measure
    p_measure = subparsers.add_parser("measure", help="Measure semantic change for a single word")
    p_measure.add_argument("word", type=str, help="Word to measure (e.g. gay, mouse, apple, cell)")
    p_measure.add_argument("--corpus", type=str, default="en-2019", help="Google Books corpus (default: en-2019)")
    p_measure.add_argument("--json", action="store_true", help="Output raw JSON data")

    # Compare
    p_compare = subparsers.add_parser("compare", help="Compare semantic change across multiple words")
    p_compare.add_argument("words", nargs="+", help="List of words to compare")
    p_compare.add_argument("--corpus", type=str, default="en-2019", help="Google Books corpus")

    # SemEval
    subparsers.add_parser("semeval", help="Run SemEval-2020 Task 1 benchmark evaluation")

    # Laws
    subparsers.add_parser("laws", help="Test statistical laws of semantic change")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Launch interactive web studio")
    p_serve.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        cmd_serve(args)
    else:
        analyzer = DiachronicAnalyzer()
        if args.command == "measure":
            cmd_measure(args, analyzer)
        elif args.command == "compare":
            cmd_compare(args, analyzer)
        elif args.command == "semeval":
            cmd_semeval(args, analyzer)
        elif args.command == "laws":
            cmd_laws(args, analyzer)


if __name__ == "__main__":
    main()
