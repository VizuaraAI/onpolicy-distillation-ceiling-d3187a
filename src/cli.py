"""Single entry point for the whole experiment.

    python -m src.cli battery   --config configs/default.yaml
    python -m src.cli audit     --config configs/default.yaml
    python -m src.cli train     --config configs/default.yaml --arm onpolicy_distill
    python -m src.cli eval      --config configs/default.yaml --run runs/onpolicy_distill
    python -m src.cli diff      --config configs/default.yaml --run runs/onpolicy_distill

Do not add top-level scripts that bypass this module. Every run writes
its resolved config and git SHA to ``meta.json`` so a result can always
be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config, override

ARMS = ("offpolicy_kd", "onpolicy_distill", "rlvr", "continued_pretrain")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_battery(cfg, args) -> int:
    """Smoke test: generate every capability and check the verifiers.

    Runs on CPU in seconds. Do this before touching a GPU.
    """
    from .data.capabilities import from_config

    ok = True
    for capability in from_config(cfg):
        try:
            examples = capability.sample(8, seed=cfg.get("seed", 0))
        except ImportError as exc:
            print(f"  {capability.name:<28} SKIP ({exc})")
            continue

        self_score = sum(
            capability.verify(ex.answer, ex.answer) for ex in examples
        ) / len(examples)
        variables = sorted({k for ex in examples for k in ex.variables})
        status = "ok" if self_score == 1.0 else "VERIFIER BUG"
        if self_score != 1.0:
            ok = False
        print(f"  {capability.name:<28} {capability.expected_group:<12} "
              f"{status:<13} variables={variables}")
        print(f"      example: {examples[0].prompt!r} -> {examples[0].answer!r}")

    if not ok:
        print("\nAt least one verifier does not accept its own gold answer. "
              "Fix that before running anything else.", file=sys.stderr)
    return 0 if ok else 1


def cmd_audit(cfg, args) -> int:
    """Score representability on the base student and freeze the split."""
    from .audit.representability import audit_capability, write_audit
    from .data.capabilities import from_config
    from .models.registry import load_model

    student = load_model(
        cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"),
    )
    audits = [
        audit_capability(student, capability, cfg,
                         n_examples=int(cfg.get("battery.n_eval_per_capability", 500)),
                         seed=int(cfg.get("seed", 0)))
        for capability in from_config(cfg)
    ]
    for audit in audits:
        print(f"  {audit.name:<28} probe={audit.probe_score:.3f} "
              f"coverage={audit.coverage_score:.3f} "
              f"R={audit.representability:.3f} "
              f"(prior: {audit.expected_group})")
        if audit.leaked_variables:
            print(f"      LEAK WARNING on {audit.leaked_variables}")

    path = cfg.run_dir("audit") / "representability.json"
    record = write_audit(path, cfg, audits)
    print(f"\nFrozen split written to {path}:")
    print(json.dumps(record["split"], indent=2))
    print("\nCommit this file. Do not regenerate it after training starts.")
    return 0


def cmd_train(cfg, args) -> int:
    """Run one training arm over the frozen split."""
    from .audit.representability import read_audit
    from .data.capabilities import select
    from .models.registry import load_model, load_student_and_teacher
    from .train.common import prepare_run_dir, write_meta

    audit = read_audit(cfg.run_dir("audit") / "representability.json")
    names = audit["split"]["represented"] + audit["split"]["absent"]
    capabilities = select(args.capabilities or names)

    run_dir = prepare_run_dir(Path(cfg.get("output_root", "runs")), args.arm)
    write_meta(run_dir, cfg, {"arm": args.arm, "audit": audit["audit_version"]})

    if args.arm == "continued_pretrain":
        from .train import continued_pretrain

        student = load_model(
            cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
            device=cfg.get("models.device", "cuda"), train=True,
        )
        results = [
            continued_pretrain.train(student, capability, cfg, seed=seed,
                                     run_dir=run_dir)
            for capability in capabilities
            for seed in cfg.get("seeds", [0])
        ]
    elif args.arm == "rlvr":
        from .train import rlvr

        student = load_model(
            cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
            device=cfg.get("models.device", "cuda"), train=True,
        )
        reference = load_model(
            cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
            device=cfg.get("models.device", "cuda"),
        )
        results = [
            rlvr.train(student, reference, capability, cfg, seed=seed,
                       run_dir=run_dir)
            for capability in capabilities
            for seed in cfg.get("seeds", [0])
        ]
    else:
        module = (__import__(f"src.train.{args.arm}", fromlist=["train"]))
        student, teacher = load_student_and_teacher(cfg)
        results = [
            module.train(student, teacher, capability, cfg, seed=seed,
                         run_dir=run_dir)
            for capability in capabilities
            for seed in cfg.get("seeds", [0])
        ]

    skipped = [r for r in results if r.skipped]
    print(f"\n{args.arm}: {len(results) - len(skipped)} runs completed, "
          f"{len(skipped)} skipped.")
    for result in skipped:
        print(f"  SKIPPED {result.capability}: {result.skipped_reason}")
    print(f"Artifacts in {run_dir}")
    return 0


def cmd_eval(cfg, args) -> int:
    """pass@1 and pass@k for a trained run against the base student."""
    from .data.capabilities import from_config
    from .eval.passk import coverage_signature, evaluate
    from .models.registry import load_model

    base = load_model(
        cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"),
    )
    trained = load_model(
        args.run, dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"),
    ) if args.run else base

    report = {}
    for capability in from_config(cfg):
        base_scores = evaluate(base, capability, cfg, seed=int(cfg.get("seed", 0)))
        trained_scores = evaluate(trained, capability, cfg,
                                  seed=int(cfg.get("seed", 0)))
        report[capability.name] = {
            "base": base_scores, "trained": trained_scores,
            "signature": coverage_signature(base_scores, trained_scores),
        }
        print(f"  {capability.name:<28} "
              f"pass@1 {base_scores['pass@1']:.3f} -> "
              f"{trained_scores['pass@1']:.3f}  "
              f"[{report[capability.name]['signature'].get('verdict', '')}]")

    out = Path(args.run or ".") / "passk.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


def cmd_diff(cfg, args) -> int:
    """Crosscoder diff of a trained run against the base student."""
    from .data.capabilities import from_config
    from .eval.diffing import diff_models, write_report
    from .models.registry import load_model

    base = load_model(
        cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"),
    )
    trained = load_model(
        args.run, dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"),
    )

    stats_by_capability = {}
    for capability in from_config(cfg):
        result = diff_models(base, trained, capability, cfg,
                             seed=int(cfg.get("seed", 0)))
        stats_by_capability[capability.name] = result["stats"]

    path = write_report(Path(args.run), stats_by_capability)
    print(f"\nWrote {path}")
    print("Report nlf_corrected in the paper, with nlf_raw beside it.")
    return 0


def _common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config key, e.g. --set budget.lr=3e-5")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    # The common flags live on both the top-level parser and every
    # subparser, so `cli --config X battery` and `cli battery --config X`
    # both work. Argparse otherwise rejects the second form, which is
    # the one the README and the run scripts use.
    common = _common_args(argparse.ArgumentParser(add_help=False))
    parser = _common_args(
        argparse.ArgumentParser(prog="src.cli", description=__doc__)
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("battery", parents=[common],
                          help="smoke test the capability battery")
    subparsers.add_parser("audit", parents=[common],
                          help="score representability, freeze split")

    train_parser = subparsers.add_parser("train", parents=[common],
                                         help="run one training arm")
    train_parser.add_argument("--arm", required=True, choices=ARMS)
    train_parser.add_argument("--capabilities", nargs="*", default=None)

    eval_parser = subparsers.add_parser("eval", parents=[common],
                                        help="pass@1 and pass@k")
    eval_parser.add_argument("--run", default=None,
                             help="path or hub id of the trained student")

    diff_parser = subparsers.add_parser("diff", parents=[common],
                                        help="crosscoder model diff")
    diff_parser.add_argument("--run", required=True)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = override(load_config(args.config), args.set)

    handlers = {
        "battery": cmd_battery, "audit": cmd_audit, "train": cmd_train,
        "eval": cmd_eval, "diff": cmd_diff,
    }
    return handlers[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
