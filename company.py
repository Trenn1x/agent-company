#!/usr/bin/env python3
"""Local corporate-agent operating simulator.

The first version is deterministic on purpose: it proves the workflow, artifact
shape, and handoff model before introducing live LLM calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import tempfile
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "i",
    "in",
    "into",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "try",
    "with",
}

TASK_STATUSES = ("open", "closed", "carried", "killed")


@dataclass(frozen=True)
class Role:
    role_id: str
    title: str
    reports_to: str
    mandate: str


@dataclass
class AgentMemo:
    role: Role
    summary: str
    recommendations: list[str]
    risks: list[str]
    asks: list[str]
    tasks: list[dict[str, str]] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class CompanyRun:
    objective: str
    context: str
    run_dir: Path
    memos: list[AgentMemo]
    tasks: list[dict[str, str]]
    board_packet: str
    provider: str
    memory_db: Path | None = None


@dataclass
class CompanyComparison:
    objective: str
    context: str
    comparison_dir: Path
    runs: list[CompanyRun]
    report_path: Path
    report: str
    memory_db: Path | None = None


@dataclass(frozen=True)
class ProviderContract:
    provider_id: str
    description: str
    inputs: list[str]
    outputs: list[str]
    forbidden_side_effects: list[str]


class MemoProvider:
    provider_id = "base"
    description = "Provider interface for creating role memos."

    def create_memos(
        self,
        engine: "CompanyEngine",
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> list[AgentMemo]:
        raise NotImplementedError

    def contract(self) -> ProviderContract:
        return ProviderContract(
            provider_id=self.provider_id,
            description=self.description,
            inputs=[
                "role id, title, reporting line, and mandate",
                "objective and optional context",
                "keywords, classification flags, and profile",
                "prior artifacts and memory summaries when supplied by the engine",
                "allowed tool policy",
            ],
            outputs=[
                "summary",
                "recommendations",
                "risks",
                "asks",
                "owned tasks with owner, priority, due date, and definition of done",
            ],
            forbidden_side_effects=[
                "send email",
                "post publicly",
                "purchase or subscribe",
                "delete user data",
                "expose private files or credentials",
            ],
        )


class DeterministicMemoProvider(MemoProvider):
    provider_id = "deterministic"
    description = "Local template-backed provider; no network or model calls."

    def create_memos(
        self,
        engine: "CompanyEngine",
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> list[AgentMemo]:
        return [
            engine._ceo(objective, context, keywords, flags, profile, days),
            engine._cpo(objective, keywords, flags, profile, days),
            engine._cto(objective, keywords, flags, profile, days),
            engine._cro(objective, keywords, flags, profile, days),
            engine._cfo_risk(objective, keywords, flags, profile, days),
            engine._coo(objective, keywords, flags, profile, days),
        ]


def extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Provider response did not contain a JSON object.") from None
        loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("Provider response JSON must be an object.")
    return loaded


class OllamaMemoProvider(MemoProvider):
    provider_id = "ollama"
    default_model = "mistral:latest"
    default_base_url = "http://127.0.0.1:11434"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
    ):
        self.model = model or self.default_model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.description = (
            f"Local Ollama provider using `{self.model}` at `{self.base_url}`; "
            "no cloud API calls."
        )

    def contract(self) -> ProviderContract:
        contract = super().contract()
        return ProviderContract(
            provider_id=self.provider_id,
            description=self.description,
            inputs=contract.inputs
            + [
                "Ollama model name",
                "local Ollama base URL",
                "strict JSON response schema",
            ],
            outputs=contract.outputs
            + [
                "valid JSON object for one requested role at a time",
                "one memo per local model call",
            ],
            forbidden_side_effects=contract.forbidden_side_effects
            + [
                "call non-local network endpoints",
                "invoke tools directly from model text",
            ],
        )

    def create_memos(
        self,
        engine: "CompanyEngine",
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> list[AgentMemo]:
        memos = []
        for role_id in [role_id for role_id in engine.roles if role_id != "chief_of_staff"]:
            memos.append(
                self._create_memo_for_role(
                    engine,
                    engine.roles[role_id],
                    objective,
                    context,
                    keywords,
                    flags,
                    profile,
                    days,
                )
            )
        return memos

    def _create_memo_for_role(
        self,
        engine: "CompanyEngine",
        role: Role,
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        errors: list[str] = []
        for attempt in range(2):
            prompt = self._build_role_prompt(
                role,
                objective,
                context,
                keywords,
                flags,
                profile,
                days,
                retry_feedback="; ".join(errors),
            )
            try:
                payload = extract_json_object(self._generate(prompt))
                return self._memo_from_role_payload(engine, role, payload)
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
                if attempt == 0:
                    continue

        return self._fallback_memo(
            engine, role.role_id, objective, context, keywords, flags, profile, days, errors
        )

    def _build_role_prompt(
        self,
        role: Role,
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
        retry_feedback: str = "",
    ) -> str:
        schema = {
            "role_id": role.role_id,
            "summary": "one concise paragraph",
            "recommendations": ["3-5 concrete recommendations"],
            "risks": ["2-4 risks"],
            "asks": ["1-3 asks"],
            "tasks": [
                {
                    "owner": role.title,
                    "priority": "P0",
                    "task": "clear action",
                    "due": "Day 1",
                    "definition_of_done": "measurable completion condition",
                }
            ],
        }
        lines = [
            "You are generating one structured executive role memo for a local multi-agent company engine.",
            "Return ONLY valid JSON. Do not include Markdown, commentary, or code fences.",
            "Do not claim to send email, post, buy, delete, or use external tools.",
            f"Return exactly one memo for role_id `{role.role_id}`. Do not use a title or acronym instead.",
            "",
            f"Role title: {role.title}",
            f"Reports to: {role.reports_to}",
            f"Role mandate: {role.mandate}",
            f"Sprint length: {days} days",
            f"Objective: {objective}",
            f"Context: {context or 'None'}",
            f"Keywords: {json.dumps(keywords)}",
            f"Flags: {json.dumps(flags, sort_keys=True)}",
            f"Profile: {json.dumps(profile, sort_keys=True)}",
        ]
        if retry_feedback:
            lines.extend(["", f"Previous invalid response reason: {retry_feedback}"])
        lines.extend(["", "Required JSON shape:", json.dumps(schema, indent=2)])
        return "\n".join(lines)

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach local Ollama at {self.base_url}. "
                "Start Ollama or use `--provider deterministic`."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON transport response.") from exc

        text = result.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama response did not include a non-empty `response` field.")
        return text

    def _memo_from_role_payload(
        self,
        engine: "CompanyEngine",
        expected_role: Role,
        payload: dict[str, object],
    ) -> AgentMemo:
        raw = payload.get("memo") if isinstance(payload.get("memo"), dict) else payload
        if not isinstance(raw, dict):
            raise ValueError("Role provider response must be a JSON object.")

        role_id = resolve_role_id(raw.get("role_id"), engine.roles)
        if role_id != expected_role.role_id:
            raise ValueError(
                f"Expected role_id `{expected_role.role_id}` but got `{raw.get('role_id')}`."
            )

        summary = coerce_text(raw.get("summary"))
        if not summary:
            raise ValueError(f"Role `{expected_role.role_id}` response is missing summary.")

        tasks = coerce_tasks(raw.get("tasks"), expected_role.title)
        return AgentMemo(
            role=expected_role,
            summary=summary,
            recommendations=coerce_list(raw.get("recommendations")),
            risks=coerce_list(raw.get("risks")),
            asks=coerce_list(raw.get("asks")),
            tasks=tasks,
        )

    def _fallback_memo(
        self,
        engine: "CompanyEngine",
        role_id: str,
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
        errors: list[str],
    ) -> AgentMemo:
        fallback_memos = DeterministicMemoProvider().create_memos(
            engine,
            objective,
            context,
            keywords,
            flags,
            profile,
            days,
        )
        for memo in fallback_memos:
            if memo.role.role_id == role_id:
                reason = errors[-1] if errors else "unknown provider error"
                memo.summary = (
                    "Ollama role memo fallback used after validation failure "
                    f"({reason}). {memo.summary}"
                )
                memo.fallback_used = True
                return memo
        raise ValueError(f"No deterministic fallback memo for role: {role_id}")

    def _memos_from_payload(self, engine: "CompanyEngine", payload: dict[str, object]) -> list[AgentMemo]:
        raw_memos = payload.get("memos")
        if not isinstance(raw_memos, list):
            raise ValueError("Provider response must include a `memos` array.")

        required_roles = [
            role_id for role_id in engine.roles if role_id != "chief_of_staff"
        ]
        memo_by_role: dict[str, AgentMemo] = {}
        for raw in raw_memos:
            if not isinstance(raw, dict):
                continue
            role_id = resolve_role_id(raw.get("role_id"), engine.roles)
            if role_id not in required_roles or role_id in memo_by_role:
                continue
            memo_by_role[role_id] = self._memo_from_role_payload(engine, engine.roles[role_id], raw)

        missing = [role_id for role_id in required_roles if role_id not in memo_by_role]
        if missing:
            fallback_flags = {
                "local_smb": False,
                "agentic": False,
                "job": False,
                "software": False,
                "revenue": False,
            }
            fallback_memos = DeterministicMemoProvider().create_memos(
                engine,
                "Provider fallback",
                "",
                [],
                fallback_flags,
                build_profile("Provider fallback", [], fallback_flags, 7),
                7,
            )
            for memo in fallback_memos:
                if memo.role.role_id in missing:
                    memo.summary = (
                        "Ollama role memo fallback used after validation failure; "
                        f"deterministic fallback used. {memo.summary}"
                    )
                    memo.fallback_used = True
                    memo_by_role[memo.role.role_id] = memo
        return [memo_by_role[role_id] for role_id in required_roles]


def resolve_role_id(value: object, roles: dict[str, Role]) -> str:
    text = coerce_text(value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    aliases = {
        "ceo": "ceo",
        "chief_executive_officer": "ceo",
        "cpo": "cpo",
        "chief_product_and_market_officer": "cpo",
        "chief_product_officer": "cpo",
        "cto": "cto",
        "chief_technology_officer": "cto",
        "cro": "cro",
        "chief_revenue_officer": "cro",
        "cfo": "cfo_risk",
        "cfo_risk": "cfo_risk",
        "cfo_and_risk_lead": "cfo_risk",
        "chief_financial_officer": "cfo_risk",
        "chief_financial_officer_and_risk_lead": "cfo_risk",
        "coo": "coo",
        "chief_operating_officer": "coo",
        "chief_of_staff": "chief_of_staff",
    }
    if normalized in roles:
        return normalized
    return aliases.get(normalized, normalized)


def coerce_text(value: object, fallback: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def coerce_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = [coerce_text(item) for item in value]
    else:
        items = [coerce_text(value)] if value is not None else []
    return [item for item in items if item] or ["No item returned."]


def coerce_tasks(value: object, default_owner: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [
            task(
                default_owner,
                "P1",
                "Review provider output and add an owned task.",
                "Day 1",
                "A concrete task is written and accepted.",
            )
        ]
    tasks: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        tasks.append(
            task(
                coerce_text(raw.get("owner"), default_owner),
                coerce_text(raw.get("priority"), "P1"),
                coerce_text(raw.get("task"), "Review provider output."),
                coerce_text(raw.get("due"), "Day 1"),
                coerce_text(
                    raw.get("definition_of_done"),
                    "The task has a measurable definition of done.",
                ),
            )
        )
    return tasks or coerce_tasks(None, default_owner)


PROVIDERS: dict[str, MemoProvider] = {
    DeterministicMemoProvider.provider_id: DeterministicMemoProvider(),
    OllamaMemoProvider.provider_id: OllamaMemoProvider(),
}


ROLES = [
    Role(
        "ceo",
        "CEO",
        "Board",
        "Set the commercial thesis, success bar, and decision discipline.",
    ),
    Role(
        "cpo",
        "Chief Product and Market Officer",
        "CEO",
        "Define the customer, pain, MVP, proof points, and scope boundaries.",
    ),
    Role(
        "cto",
        "Chief Technology Officer",
        "CEO",
        "Translate the product wedge into a buildable architecture and delivery plan.",
    ),
    Role(
        "cro",
        "Chief Revenue Officer",
        "CEO",
        "Turn the wedge into a pipeline, offer, demo path, and conversion script.",
    ),
    Role(
        "cfo_risk",
        "CFO and Risk Lead",
        "CEO",
        "Protect time, capital, legal exposure, and opportunity cost.",
    ),
    Role(
        "coo",
        "Chief Operating Officer",
        "CEO",
        "Convert strategy into cadence, owners, deadlines, and operating constraints.",
    ),
    Role(
        "chief_of_staff",
        "Chief of Staff",
        "CEO",
        "Reconcile memos into the board packet, task list, and decision log.",
    ),
]


def slugify(value: str, fallback: str = "run") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or fallback


def wrap(text: str, width: int = 88) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
    seen: set[str] = set()
    results: list[str] = []
    for word in words:
        if word in STOP_WORDS or word in seen:
            continue
        seen.add(word)
        results.append(word)
        if len(results) >= limit:
            break
    return results


def classify_objective(objective: str) -> dict[str, bool]:
    lower = objective.lower()
    return {
        "local_smb": any(
            token in lower
            for token in [
                "smb",
                "small business",
                "service business",
                "charleston",
                "hvac",
                "plumbing",
                "roof",
                "restaurant",
                "dental",
            ]
        ),
        "agentic": any(token in lower for token in ["agent", "automation", "autopilot", "workflow"]),
        "job": any(token in lower for token in ["job", "resume", "application", "contract", "gig"]),
        "software": any(
            token in lower for token in ["app", "software", "mvp", "tool", "platform", "cli"]
        ),
        "revenue": any(
            token in lower
            for token in ["sell", "sales", "client", "buyer", "revenue", "roi", "money"]
        ),
    }


def build_profile(
    objective: str, keywords: list[str], flags: dict[str, bool], days: int
) -> dict[str, object]:
    if flags["local_smb"]:
        return {
            "segment": "Charleston-area owner-operated service businesses with urgent inbound leads",
            "buyer": "Owner, general manager, or office manager",
            "pain": "missed calls, slow follow-up, manual quote tracking, and no clean view of stale leads",
            "mvp": "a lead-capture and follow-up command board using a realistic service-business workflow",
            "offer": "$500-$1,500 fixed-fee pilot: set up one workflow, one dashboard, and one follow-up sequence",
            "success_metric": (
                f"Within {days} days, get one qualified Charleston-area owner to agree to a paid pilot "
                "or a clearly scoped 15-minute pilot review call."
            ),
            "proof_signal": "owner asks for pricing, books a review call, or provides real workflow data",
            "demo_script_title": "Missed lead recovery workflow",
            "prospect_archetypes": [
                "HVAC/plumbing company",
                "roofing or storm restoration company",
                "property management office",
                "dental or elective-care clinic",
                "restaurant/catering operator",
            ],
            "keywords": keywords,
        }

    if flags["job"]:
        return {
            "segment": "hiring managers seeking practical automation builders",
            "buyer": "Hiring manager or referral contact",
            "pain": "they need proof that you can reduce operational labor quickly",
            "mvp": "a role-specific work sample that looks useful on day one",
            "offer": "proof-of-work packet plus targeted application follow-up",
            "success_metric": (
                f"Within {days} days, earn one recruiter/hiring-manager reply or one referral-backed "
                "conversation from a targeted application."
            ),
            "proof_signal": "reply, referral, interview request, or request for more work samples",
            "demo_script_title": "Week-one automation artifact",
            "prospect_archetypes": [
                "forward-deployed AI team",
                "operations automation team",
                "revenue operations team",
                "security automation team",
                "customer education automation team",
            ],
            "keywords": keywords,
        }

    if flags["agentic"]:
        return {
            "segment": "you as the first operator across job search, SMB sales, and build projects",
            "buyer": "internal operator first; external buyer only after internal proof",
            "pain": "too many promising directions without a forced execution cadence",
            "mvp": "a corporate-agent run folder that creates decisions, tasks, thresholds, and review cadence",
            "offer": "operator-in-a-box that converts vague goals into owned execution loops",
            "success_metric": (
                f"Within {days} days, complete one full operating loop with a documented go/pivot/kill decision."
            ),
            "proof_signal": "a closed task list and a decision based on external or operational evidence",
            "demo_script_title": "Corporate-agent operating loop",
            "internal_first": True,
            "next_step": "run one real objective through the operating loop, close/carry/kill tasks, and write the daily review",
            "proof_actions": [
                "Run the company on one real objective.",
                "Update at least three task statuses with notes.",
                "Produce one daily executive review artifact.",
                "Record a go, pivot, or kill decision.",
                "Document what an LLM provider must supply later.",
            ],
            "prospect_archetypes": [
                "solo operator",
                "fractional automation consultant",
                "small agency owner",
                "job seeker with proof-of-work strategy",
                "local SMB builder",
            ],
            "keywords": keywords,
        }

    return {
        "segment": "the smallest reachable user group implied by the objective",
        "buyer": "person who owns the painful workflow",
        "pain": "one expensive or annoying workflow they already recognize",
        "mvp": "the smallest demo that makes the before and after obvious",
        "offer": "small paid or time-boxed validation project",
        "success_metric": f"Within {days} days, obtain one concrete proof signal from a reachable user.",
        "proof_signal": "useful feedback, committed next step, real data, or paid validation",
        "demo_script_title": "Before-and-after workflow demo",
        "prospect_archetypes": [
            "reachable buyer",
            "domain operator",
            "current user",
            "referral source",
            "expert reviewer",
        ],
        "keywords": keywords,
    }


def task(owner: str, priority: str, description: str, due: str, done: str) -> dict[str, str]:
    return {
        "owner": owner,
        "priority": priority,
        "task": description,
        "due": due,
        "definition_of_done": done,
        "status": "open",
        "status_note": "",
        "updated_at": "",
    }


def normalize_tasks(tasks: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for index, item in enumerate(tasks, start=1):
        copy = {key: str(value) for key, value in item.items()}
        copy.setdefault("id", f"T{index:02d}")
        copy.setdefault("owner", "Unassigned")
        copy.setdefault("priority", "P1")
        copy.setdefault("task", "Untitled task")
        copy.setdefault("due", "TBD")
        copy.setdefault("definition_of_done", "Done when explicitly closed.")
        status = copy.get("status", "open").lower()
        copy["status"] = status if status in TASK_STATUSES else "open"
        copy.setdefault("status_note", "")
        copy.setdefault("updated_at", "")
        normalized.append(copy)
    return normalized


def canonical_task_id(value: str) -> str:
    raw = value.strip().upper()
    if raw.isdigit():
        return f"T{int(raw):02d}"
    match = re.fullmatch(r"T(\d+)", raw)
    if match:
        return f"T{int(match.group(1)):02d}"
    return raw


class CompanyEngine:
    def __init__(
        self,
        roles: Iterable[Role] = ROLES,
        provider: str | MemoProvider = "deterministic",
        provider_model: str | None = None,
        ollama_url: str | None = None,
    ):
        self.roles = {role.role_id: role for role in roles}
        if isinstance(provider, str):
            if provider not in PROVIDERS:
                raise ValueError(f"Unknown provider: {provider}")
            if provider == OllamaMemoProvider.provider_id:
                self.provider = OllamaMemoProvider(
                    model=provider_model,
                    base_url=ollama_url,
                )
            else:
                self.provider = PROVIDERS[provider]
        else:
            self.provider = provider

    def run(
        self,
        objective: str,
        context: str = "",
        run_name: str | None = None,
        runs_dir: Path | str = "runs",
        days: int = 7,
        memory_db: Path | str | None = None,
        use_memory: bool = True,
    ) -> CompanyRun:
        objective = objective.strip()
        if not objective:
            raise ValueError("Objective is required.")

        run_dir = self._prepare_run_dir(objective, run_name, Path(runs_dir))
        keywords = extract_keywords(objective)
        flags = classify_objective(objective)
        profile = build_profile(objective, keywords, flags, days)

        memos = self.provider.create_memos(
            self, objective, context, keywords, flags, profile, days
        )
        tasks = [item for memo in memos for item in memo.tasks]
        chief = self._chief_of_staff(objective, keywords, flags, profile, days, memos, tasks)
        memos.append(chief)
        tasks.extend(chief.tasks)
        tasks = normalize_tasks(tasks)

        board_packet = render_board_packet(objective, context, memos, tasks, profile, days)
        self._write_artifacts(run_dir, objective, context, memos, tasks, board_packet, profile)

        resolved_memory_db = None
        if use_memory:
            resolved_memory_db = Path(memory_db) if memory_db else default_memory_db(run_dir.parent)
            sync_run_to_memory(
                run_dir,
                resolved_memory_db,
                objective=objective,
                profile=profile,
                provider_id=self.provider.provider_id,
            )

        return CompanyRun(
            objective=objective,
            context=context,
            run_dir=run_dir,
            memos=memos,
            tasks=tasks,
            board_packet=board_packet,
            provider=self.provider.provider_id,
            memory_db=resolved_memory_db,
        )

    def _prepare_run_dir(self, objective: str, run_name: str | None, runs_dir: Path) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = slugify(run_name or objective)
        run_dir = runs_dir / f"{stamp}-{name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _ceo(
        self,
        objective: str,
        context: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        wedge = "single paid validation sprint" if flags["revenue"] else "single proof sprint"
        if flags["local_smb"]:
            wedge = "one local SMB pain, one demo, one owner conversation"
        elif flags["job"]:
            wedge = "one target role class, one proof artifact, one application packet"
        elif flags["agentic"]:
            wedge = "one operating loop with explicit human approval gates"

        return AgentMemo(
            role=self.roles["ceo"],
            summary=(
                f"Treat the objective as a {days}-day experiment, not a sprawling venture. "
                f"The operating thesis is: {wedge}."
            ),
            recommendations=[
                "Pick one buyer or user segment before building anything larger.",
                "Define the proof as a concrete behavior: booked call, usable demo, shipped artifact, or paid pilot.",
                "Keep the first run manual where judgment matters and automated where repetition is obvious.",
                "Stop expanding scope until one complete loop has been run end to end.",
            ],
            risks=[
                "A corporate-agent setup can create impressive planning output without forcing market contact.",
                "Too many roles can diffuse accountability unless every memo ends in owned tasks.",
                "The first version should avoid autonomous external action without explicit human approval.",
            ],
            asks=[
                "Approve one objective for the first company run.",
                "Confirm whether the output should optimize for cash, job leverage, or reusable IP.",
            ],
            tasks=[
                task(
                    "CEO",
                    "P0",
                    "Approve or edit the generated sprint success metric.",
                    "Day 0",
                    "board_packet.md has one measurable success metric and no competing primary goal.",
                )
            ],
        )

    def _cpo(
        self,
        objective: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        segment = str(profile["segment"])
        pain = str(profile["pain"])
        mvp = str(profile["mvp"])

        return AgentMemo(
            role=self.roles["cpo"],
            summary=f"Product should focus on {segment}. The first pain to test is {pain}.",
            recommendations=[
                f"Use keywords as positioning anchors: {', '.join(keywords) or 'objective-specific terms'}.",
                f"Build {mvp}.",
                "Make the demo explainable in under two minutes.",
                "Write acceptance criteria before implementation starts.",
            ],
            risks=[
                "A broad product promise will produce vague feedback.",
                "If the demo requires too much explanation, the pain is probably not sharp enough.",
            ],
            asks=[
                "Choose the exact first user or buyer archetype.",
                "Provide one real example document, workflow, inbox thread, or dataset if available.",
            ],
            tasks=[
                task(
                    "CPO",
                    "P0",
                    "Review and tighten product_brief.md.",
                    "Day 1",
                    "ICP, pain, promise, non-goals, MVP, and acceptance criteria are specific enough to build from.",
                ),
                task(
                    "CPO",
                    "P1",
                    "Turn demo_script.md into a two-minute walkthrough.",
                    "Day 2",
                    "The script names the before state, after state, proof moment, and next ask.",
                ),
            ],
        )

    def _cto(
        self,
        objective: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        if flags["agentic"]:
            architecture = "file-backed roles, deterministic templates, run folders, and an LLM provider boundary"
            tasks = [
                task(
                    "CTO",
                    "P0",
                    "Ship the task lifecycle and daily review commands.",
                    "Day 1",
                    "`status`, `task`, and `review` commands work against an existing run folder.",
                ),
                task(
                    "CTO",
                    "P1",
                    "Document the deterministic v1 and future LLM provider boundary.",
                    "Day 2",
                    "technical_plan.md or README explains what stays deterministic and where provider-backed agents plug in.",
                ),
            ]
        elif flags["software"]:
            architecture = "thin web or CLI surface, local persistence first, and deployable static artifacts where possible"
            tasks = [
                task(
                    "CTO",
                    "P0",
                    "Build or adapt the demo artifact described in product_brief.md.",
                    "Day 3",
                    "There is a clickable, runnable, or inspectable demo artifact that supports the script.",
                ),
                task(
                    "CTO",
                    "P1",
                    "Document the smallest data/input needed from a real prospect.",
                    "Day 3",
                    "technical_plan.md lists required fields, sample data, setup assumptions, and constraints.",
                ),
            ]
        else:
            architecture = "documents, scripts, and a repeatable checklist before custom infrastructure"
            tasks = [
                task(
                    "CTO",
                    "P0",
                    "Build or adapt the demo artifact described in product_brief.md.",
                    "Day 3",
                    "There is a clickable, runnable, or inspectable demo artifact that supports the script.",
                ),
                task(
                    "CTO",
                    "P1",
                    "Document the smallest data/input needed from a real prospect.",
                    "Day 3",
                    "technical_plan.md lists required fields, sample data, setup assumptions, and constraints.",
                ),
            ]

        return AgentMemo(
            role=self.roles["cto"],
            summary=f"Technical approach: {architecture}. Build for auditability before autonomy.",
            recommendations=[
                "Use repo-visible artifacts so every agent handoff can be inspected.",
                "Represent tasks as structured JSON as well as Markdown.",
                "Keep external side effects behind commands that require explicit approval.",
                "Make every run reproducible enough to compare decisions over time.",
            ],
            risks=[
                "LLM orchestration without state discipline will be hard to debug.",
                "Autonomous email, posting, or purchasing should not be part of v1.",
                "A pretty UI is lower leverage than a reliable run ledger at this stage.",
            ],
            asks=[
                "Decide which tools the agents may use in a later version: browser, Gmail, GitHub, shell, or none.",
                "Decide where durable memory should live: Markdown, SQLite, or both.",
            ],
            tasks=tasks,
        )

    def _cro(
        self,
        objective: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        if flags["local_smb"]:
            channel = "10 targeted owner/operator messages with a concrete workflow demo"
            tasks = [
                task(
                    "CRO",
                    "P0",
                    "Replace placeholders in outreach_plan.md with 10 real targets.",
                    "Day 2",
                    "Each row has company, contact path, reason to contact, message variant, and follow-up date.",
                ),
                task(
                    "CRO",
                    "P1",
                    "Start objection_log.md from the first responses or non-responses.",
                    "Day 3",
                    "Every response can be categorized as interest, objection, referral, or no fit.",
                ),
            ]
        elif flags["job"]:
            channel = "targeted applications plus proof-of-work follow-up"
            tasks = [
                task(
                    "CRO",
                    "P0",
                    "Replace placeholders in outreach_plan.md with 10 real targets.",
                    "Day 2",
                    "Each row has company, contact path, reason to contact, message variant, and follow-up date.",
                ),
                task(
                    "CRO",
                    "P1",
                    "Start objection_log.md from the first responses or non-responses.",
                    "Day 3",
                    "Every response can be categorized as interest, objection, referral, or no fit.",
                ),
            ]
        elif flags["agentic"]:
            channel = "internal use first, then public build-in-public proof if it works"
            tasks = [
                task(
                    "CRO",
                    "P0",
                    "Define the first internal proof loop in outreach_plan.md.",
                    "Day 2",
                    "outreach_plan.md names the internal scenario, proof evidence, user promise, and follow-up decision.",
                ),
                task(
                    "CRO",
                    "P1",
                    "Start objection_log.md as an operator-friction log.",
                    "Day 3",
                    "Every friction point is categorized as unclear value, workflow drag, missing capability, or no fit.",
                ),
            ]
        else:
            channel = "five direct conversations with people close to the pain"
            tasks = [
                task(
                    "CRO",
                    "P0",
                    "Replace placeholders in outreach_plan.md with 10 real targets.",
                    "Day 2",
                    "Each row has company, contact path, reason to contact, message variant, and follow-up date.",
                ),
                task(
                    "CRO",
                    "P1",
                    "Start objection_log.md from the first responses or non-responses.",
                    "Day 3",
                    "Every response can be categorized as interest, objection, referral, or no fit.",
                ),
            ]
        offer = str(profile["offer"])

        return AgentMemo(
            role=self.roles["cro"],
            summary=f"Revenue motion should start with {channel}. The offer is: {offer}.",
            recommendations=[
                "Do not wait for a finished product before asking for feedback.",
                "Lead with the specific pain and artifact, not AI as a generic capability.",
                "Track each prospect, objection, next step, and conversion status.",
                "Use a short call to action: reply with one workflow, book a 15-minute review, or approve a pilot.",
            ],
            risks=[
                "Cold outreach fails when it sounds like a capability list instead of a solved problem.",
                "A free demo can become unpaid consulting unless the next step is explicit.",
            ],
            asks=[
                "Confirm the first outreach audience.",
                "Confirm whether the first ask should be feedback, call booking, or paid pilot.",
            ],
            tasks=tasks,
        )

    def _cfo_risk(
        self,
        objective: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        return AgentMemo(
            role=self.roles["cfo_risk"],
            summary=(
                "The experiment should preserve cash and attention. Spend time only where it "
                "creates proof, pipeline, or reusable operating assets."
            ),
            recommendations=[
                "Set a hard timebox before build work starts.",
                "Use free/local tooling until there is a buyer, interview, or clear productivity gain.",
                "Define kill criteria now so the project cannot survive on novelty alone.",
                "Separate internal automation experiments from anything that creates legal, financial, or reputational exposure.",
            ],
            risks=[
                "Opportunity cost is the main cost; more ideas can crowd out the best one.",
                "Revenue projections before buyer contact should be treated as assumptions.",
                "External messaging by agents needs human review.",
            ],
            asks=[
                "Set max hours for this sprint.",
                "Choose the smallest allowed budget, ideally zero dollars for v1.",
            ],
            tasks=[
                task(
                    "CFO/Risk",
                    "P0",
                    "Approve validation_thresholds.md.",
                    "Day 1",
                    "Go, pivot, kill, budget, and approval thresholds are explicit and measurable.",
                )
            ],
        )

    def _coo(
        self,
        objective: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
    ) -> AgentMemo:
        return AgentMemo(
            role=self.roles["coo"],
            summary="Run the company as a tight weekly operating cadence with daily artifacts.",
            recommendations=[
                "Start each run with one objective and one success metric.",
                "Every role produces a memo with recommendations, risks, asks, and tasks.",
                "Chief of Staff reconciles conflicts and produces the board packet.",
                "Review the task list daily; close, carry, or kill each item.",
            ],
            risks=[
                "The process can become theater if tasks are not closed.",
                "Role debates should expose tradeoffs, not create more documents for their own sake.",
            ],
            asks=[
                "Use one run folder as the source of truth.",
                "Keep a short decision log after every human override.",
            ],
            tasks=[
                task(
                    "COO",
                    "P0",
                    "Run the cadence in sprint_calendar.md.",
                    "Day 1",
                    "Daily review, build block, outreach block, and decision checkpoint are scheduled.",
                ),
                task(
                    "COO",
                    "P1",
                    "Review task status at the end of each day.",
                    "Daily",
                    "Tasks are marked closed, carried, or killed with one-line rationale.",
                ),
            ],
        )

    def _chief_of_staff(
        self,
        objective: str,
        keywords: list[str],
        flags: dict[str, bool],
        profile: dict[str, object],
        days: int,
        memos: list[AgentMemo],
        tasks: list[dict[str, str]],
    ) -> AgentMemo:
        return AgentMemo(
            role=self.roles["chief_of_staff"],
            summary=(
                "The executive team converges on a narrow, testable sprint. The board packet "
                "should be treated as the operating contract for the next run."
            ),
            recommendations=[
                "Approve the sprint only if the success metric is measurable.",
                "Use the first run to validate the workflow and the business thesis simultaneously.",
                "Move from planning to external signal as early as possible.",
                "Keep all future improvements tied to observed bottlenecks in this run.",
            ],
            risks=[
                "The team has a bias toward building; CRO and CFO/Risk should force outside signal.",
                "The CEO must choose between cash, job leverage, and reusable IP if they conflict.",
            ],
            asks=[
                "Approve or revise the board packet.",
                "Select the first task owner to unblock Day 1.",
            ],
            tasks=[
                task(
                    "Chief of Staff",
                    "P0",
                    "Review the generated operating packet for conflicts.",
                    "Day 0",
                    "Board packet, product brief, outreach plan, thresholds, calendar, and tasks agree with each other.",
                )
            ],
        )

    def _write_artifacts(
        self,
        run_dir: Path,
        objective: str,
        context: str,
        memos: list[AgentMemo],
        tasks: list[dict[str, str]],
        board_packet: str,
        profile: dict[str, object],
    ) -> None:
        (run_dir / "00_objective.md").write_text(
            render_objective(objective, context), encoding="utf-8"
        )
        for index, memo in enumerate(memos, start=1):
            (run_dir / f"{index:02d}_{memo.role.role_id}_memo.md").write_text(
                render_memo(memo), encoding="utf-8"
            )
        (run_dir / "board_packet.md").write_text(board_packet, encoding="utf-8")
        (run_dir / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
        (run_dir / "tasks.md").write_text(render_tasks(tasks), encoding="utf-8")
        (run_dir / "run_status.md").write_text(render_status_report(run_dir, tasks), encoding="utf-8")
        (run_dir / "decision_log.md").write_text(
            render_decision_log(objective, profile), encoding="utf-8"
        )
        (run_dir / "product_brief.md").write_text(
            render_product_brief(objective, profile), encoding="utf-8"
        )
        (run_dir / "demo_script.md").write_text(
            render_demo_script(profile), encoding="utf-8"
        )
        (run_dir / "outreach_plan.md").write_text(
            render_outreach_plan(profile), encoding="utf-8"
        )
        (run_dir / "validation_thresholds.md").write_text(
            render_validation_thresholds(profile), encoding="utf-8"
        )
        (run_dir / "sprint_calendar.md").write_text(
            render_sprint_calendar(profile), encoding="utf-8"
        )
        (run_dir / "technical_plan.md").write_text(
            render_technical_plan(profile), encoding="utf-8"
        )
        (run_dir / "provider_contract.md").write_text(
            render_provider_contract(self.provider.contract()), encoding="utf-8"
        )
        (run_dir / "objection_log.md").write_text(
            render_objection_log(profile), encoding="utf-8"
        )


def render_objective(objective: str, context: str) -> str:
    body = ["# Objective", "", objective.strip(), ""]
    if context.strip():
        body.extend(["## Context", "", context.strip(), ""])
    return "\n".join(body)


def bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def sentence_case(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def render_memo(memo: AgentMemo) -> str:
    return "\n".join(
        [
            f"# {memo.role.title} Memo",
            "",
            f"**Reports to:** {memo.role.reports_to}",
            "",
            f"**Mandate:** {memo.role.mandate}",
            "",
            "## Summary",
            "",
            memo.summary,
            "",
            "## Recommendations",
            "",
            bullets(memo.recommendations),
            "",
            "## Risks",
            "",
            bullets(memo.risks),
            "",
            "## Asks",
            "",
            bullets(memo.asks),
            "",
            "## Owned Tasks",
            "",
            render_tasks(memo.tasks, heading=False) if memo.tasks else "- None",
            "",
        ]
    )


def render_provider_contract(contract: ProviderContract) -> str:
    return "\n".join(
        [
            "# Provider Contract",
            "",
            f"- Provider: {contract.provider_id}",
            f"- Description: {contract.description}",
            "",
            "## Inputs",
            "",
            bullets(contract.inputs),
            "",
            "## Required Outputs",
            "",
            bullets(contract.outputs),
            "",
            "## Forbidden Side Effects",
            "",
            bullets(contract.forbidden_side_effects),
            "",
            "## Validation Rule",
            "",
            "The engine owns artifact writing, task normalization, memory sync, and approval gates. Providers only return structured memo content.",
            "",
        ]
    )


def render_tasks(tasks: list[dict[str, str]], heading: bool = True) -> str:
    lines: list[str] = []
    if heading:
        lines.extend(["# Task List", ""])
    for idx, item in enumerate(tasks, start=1):
        label = item.get("id", str(idx))
        lines.extend(
            [
                f"## {label}. {item.get('task', 'Untitled task')}",
                "",
                f"- Owner: {item.get('owner', 'Unassigned')}",
                f"- Priority: {item.get('priority', 'P1')}",
                f"- Due: {item.get('due', 'TBD')}",
                f"- Status: {item.get('status', 'open')}",
                f"- Done when: {item.get('definition_of_done', 'Done when explicitly closed.')}",
            ]
        )
        if item.get("status_note"):
            lines.append(f"- Status note: {item['status_note']}")
        if item.get("updated_at"):
            lines.append(f"- Updated: {item['updated_at']}")
        lines.append("")
    return "\n".join(lines)


def render_product_brief(objective: str, profile: dict[str, object]) -> str:
    keywords = ", ".join(profile.get("keywords", []) or ["objective-specific proof"])
    if profile.get("internal_first"):
        non_goals = [
            "- Do not turn the operating loop into a broad SaaS platform before it works for one real objective.",
            "- Do not automate external side effects without explicit human approval.",
            "- Do not add LLM providers until the deterministic run lifecycle is useful and inspectable.",
        ]
        acceptance = [
            f"- Demo shows the before state, the improved workflow, and the proof moment: {profile['proof_signal']}.",
            "- A non-technical operator can understand the value in under two minutes.",
            f"- The next step is explicit: {profile.get('next_step', 'run one full operating loop')}.",
        ]
    else:
        non_goals = [
            "- Do not build a broad platform before the first buyer conversation.",
            "- Do not automate external outreach without human approval.",
            "- Do not add integrations unless they directly support the first proof signal.",
        ]
        acceptance = [
            f"- Demo shows the before state, the improved workflow, and the proof moment: {profile['proof_signal']}.",
            "- A non-technical buyer can understand the value in under two minutes.",
            "- The next step is explicit: pilot review, pricing discussion, or real workflow-data request.",
        ]
    return "\n".join(
        [
            "# Product Brief",
            "",
            f"## Objective",
            "",
            objective,
            "",
            "## ICP",
            "",
            f"- Segment: {profile['segment']}",
            f"- Buyer: {profile['buyer']}",
            f"- Trigger pain: {profile['pain']}",
            "",
            "## Promise",
            "",
            f"Help the buyer recover or clarify value from this pain: {profile['pain']}.",
            "",
            "## MVP",
            "",
            sentence_case(profile["mvp"]),
            "",
            "## Positioning Anchors",
            "",
            keywords,
            "",
            "## Non-Goals",
            "",
            *non_goals,
            "",
            "## Acceptance Criteria",
            "",
            *acceptance,
            "",
        ]
    )


def render_demo_script(profile: dict[str, object]) -> str:
    if profile.get("internal_first"):
        return "\n".join(
            [
                f"# Demo Script: {profile['demo_script_title']}",
                "",
                "## Before",
                "",
                f"The operator has {profile['pain']}. Work starts as a vague goal, then fragments into notes, tasks, and half-finished ideas.",
                "",
                "## After",
                "",
                f"Show {profile['mvp']} as the operating loop: objective, role memos, board packet, tasks, status updates, and daily review.",
                "",
                "## Proof Moment",
                "",
                f"Point to the outcome that matters: {profile['proof_signal']}.",
                "",
                "## Ask",
                "",
                f"Ask whether the operator would use this for the next real decision: {profile.get('next_step', 'run one full operating loop')}.",
                "",
            ]
        )

    return "\n".join(
        [
            f"# Demo Script: {profile['demo_script_title']}",
            "",
            "## Before",
            "",
            f"The buyer currently deals with {profile['pain']}. Show the messy or manual version first.",
            "",
            "## After",
            "",
            f"Show {profile['mvp']} as the cleaner workflow.",
            "",
            "## Proof Moment",
            "",
            f"Point to the outcome that matters: {profile['proof_signal']}.",
            "",
            "## Ask",
            "",
            f"Ask whether they would review a scoped pilot around this offer: {profile['offer']}.",
            "",
        ]
    )


def render_outreach_plan(profile: dict[str, object]) -> str:
    if profile.get("internal_first"):
        proof_actions = list(profile.get("proof_actions", []))
        rows = []
        for idx, action in enumerate(proof_actions, start=1):
            rows.append(
                f"| {idx} | Internal operator | {action} | Run folder artifact, task update, or decision log entry | Open |"
            )
        return "\n".join(
            [
                "# Internal Proof Plan",
                "",
                "## Offer",
                "",
                str(profile["offer"]),
                "",
                "## Proof Loops",
                "",
                "| # | User | Action | Evidence To Capture | Status |",
                "|---|---|---|---|---|",
                *rows,
                "",
                "## Follow-Up Rule",
                "",
                "If the loop feels useful, keep tightening the engine. If it feels like ceremony, kill or simplify the role/process causing drag.",
                "",
            ]
        )

    archetypes = list(profile.get("prospect_archetypes", []))
    rows = []
    for idx in range(1, 11):
        archetype = archetypes[(idx - 1) % len(archetypes)] if archetypes else "target buyer"
        rows.append(
            "| {idx} | TBD | {archetype} | Website/contact form/LinkedIn/email | Likely has {pain} | "
            "Hi [Name], I am testing a small workflow demo for {archetype} teams that helps with "
            "{pain}. If I sent a two-minute example, would you tell me whether it is useful? | "
            "2 business days | Open |".format(
                idx=idx,
                archetype=archetype,
                pain=profile["pain"],
            )
        )
    return "\n".join(
        [
            "# Outreach Plan",
            "",
            f"## Offer",
            "",
            str(profile["offer"]),
            "",
            "## Target List",
            "",
            "| # | Company | Archetype | Contact Path | Reason To Contact | Message Draft | Follow-Up | Status |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            "## Follow-Up Template",
            "",
            "Hi [Name], quick follow-up. The demo is intentionally small: one workflow, one visible before/after, and no platform commitment. Worth a 15-minute look?",
            "",
        ]
    )


def render_validation_thresholds(profile: dict[str, object]) -> str:
    if profile.get("internal_first"):
        return "\n".join(
            [
                "# Validation Thresholds",
                "",
                "## Success Metric",
                "",
                str(profile["success_metric"]),
                "",
                "## Go",
                "",
                f"- The operator produces this proof signal: {profile['proof_signal']}.",
                "- The run creates less ambiguity than it adds, and the next task is obvious from the artifacts.",
                "",
                "## Pivot",
                "",
                "- The role structure is useful, but task lifecycle, memory, or provider boundaries need a different shape.",
                "- Reviews surface real friction, but the current artifacts do not make the next action obvious.",
                "- The operating loop works for one objective type but not the target mix of job search, SMB sales, and build projects.",
                "",
                "## Kill",
                "",
                "- The process creates more ceremony than decision clarity.",
                "- Tasks are not easier to close, carry, or kill after using the engine.",
                "- No useful go/pivot/kill decision appears after the sprint timebox.",
                "",
                "## Budget And Approval",
                "",
                "- Default cash budget: $0.",
                "- Human approval required before email sends, public posts, paid tools, or client commitments.",
                "",
            ]
        )

    return "\n".join(
        [
            "# Validation Thresholds",
            "",
            "## Success Metric",
            "",
            str(profile["success_metric"]),
            "",
            "## Go",
            "",
            f"- A qualified buyer/user produces this proof signal: {profile['proof_signal']}.",
            "- Or a prospect gives real workflow data and asks for a scoped next step.",
            "",
            "## Pivot",
            "",
            "- The pain is real, but the buyer segment does not respond.",
            "- Prospects engage but object to the offer, price, or workflow shape.",
            "- A different adjacent pain is repeatedly mentioned.",
            "",
            "## Kill",
            "",
            "- No reachable buyer/user can be identified.",
            "- The demo cannot be explained in two minutes.",
            "- No proof signal appears after the sprint timebox.",
            "",
            "## Budget And Approval",
            "",
            "- Default cash budget: $0.",
            "- Human approval required before email sends, public posts, paid tools, or client commitments.",
            "",
        ]
    )


def render_sprint_calendar(profile: dict[str, object]) -> str:
    if profile.get("internal_first"):
        return "\n".join(
            [
                "# Sprint Calendar",
                "",
                "## Daily Cadence",
                "",
                "- Morning: choose the one engine or operating-loop task that changes proof odds most.",
                "- Midday: build, run, or review block.",
                "- Evening: update tasks as closed, carried, or killed with one-line rationale.",
                "",
                "## Seven-Day Plan",
                "",
                "1. Day 0: approve board packet, success metric, and validation thresholds.",
                "2. Day 1: verify task lifecycle commands against a real run.",
                "3. Day 2: produce the first daily executive review.",
                "4. Day 3: run one real objective through the company loop.",
                "5. Day 4: log friction and tighten the weakest role handoff.",
                "6. Day 5: document provider-boundary requirements from observed friction.",
                "7. Day 6: run one more review using updated tasks.",
                "8. Day 7: make go, pivot, or kill decision.",
                "",
            ]
        )

    return "\n".join(
        [
            "# Sprint Calendar",
            "",
            "## Daily Cadence",
            "",
            "- Morning: choose the one task that changes the proof odds most.",
            "- Midday: build, research, or outreach block.",
            "- Evening: update tasks as closed, carried, or killed with one-line rationale.",
            "",
            "## Seven-Day Plan",
            "",
            "1. Day 0: approve board packet, success metric, and validation thresholds.",
            "2. Day 1: tighten product brief and identify 10 real targets.",
            "3. Day 2: finish demo script and outreach drafts.",
            "4. Day 3: build or adapt the smallest demo artifact.",
            "5. Day 4: send approved outreach or collect internal proof signal.",
            "6. Day 5: follow up and log objections.",
            "7. Day 6: run one review call, if booked, or revise the offer from feedback.",
            "8. Day 7: make go, pivot, or kill decision.",
            "",
        ]
    )


def render_technical_plan(profile: dict[str, object]) -> str:
    if profile.get("internal_first"):
        return "\n".join(
            [
                "# Technical Plan",
                "",
                "## Demo Artifact",
                "",
                str(profile["mvp"]),
                "",
                "## Deterministic V1",
                "",
                "- Role memos are generated from deterministic templates.",
                "- Runs are file-backed and inspectable.",
                "- Tasks have stable IDs and lifecycle statuses: open, closed, carried, killed.",
                "- Daily reviews are generated from current task state.",
                "- External side effects stay outside the engine unless a future permission layer explicitly allows them.",
                "",
                "## Future LLM Provider Boundary",
                "",
                "- A provider receives role, mandate, objective, context, prior artifacts, and allowed tools.",
                "- A provider returns structured memo fields: summary, recommendations, risks, asks, and tasks.",
                "- The engine validates returned tasks before writing artifacts.",
                "- Provider calls must not directly send email, post, buy, delete, or expose private data.",
                "",
                "## Minimum Inputs Needed",
                "",
                "- One real objective.",
                "- Optional context file.",
                "- Human approval gate for any external action.",
                "- A daily review note or decision when the run changes state.",
                "",
            ]
        )

    return "\n".join(
        [
            "# Technical Plan",
            "",
            "## Demo Artifact",
            "",
            str(profile["mvp"]),
            "",
            "## Minimum Inputs Needed",
            "",
            "- One real or realistic workflow example.",
            "- A small sample of current lead/task/customer records, sanitized if needed.",
            "- The buyer's current follow-up or decision rule.",
            "- The output they would actually review daily or weekly.",
            "",
            "## Implementation Notes",
            "",
            "- Prefer static HTML, CSV/Markdown, or local scripts for the first demo.",
            "- Avoid paid infrastructure until a buyer asks for the next step.",
            "- Keep any prospect data in the run folder or a clearly named local project folder.",
            "",
        ]
    )


def render_objection_log(profile: dict[str, object] | None = None) -> str:
    if profile and profile.get("internal_first"):
        return "\n".join(
            [
                "# Operator Friction Log",
                "",
                "| Date | Area | Friction Type | Signal | Reply / Next Step | Status |",
                "|---|---|---|---|---|---|",
                "| TBD | TBD | TBD | TBD | TBD | Open |",
                "",
            ]
        )

    return "\n".join(
        [
            "# Objection Log",
            "",
            "| Date | Company | Response Type | Objection Or Signal | Reply / Next Step | Status |",
            "|---|---|---|---|---|---|",
            "| TBD | TBD | TBD | TBD | TBD | Open |",
            "",
        ]
    )


def render_decision_log(objective: str, profile: dict[str, object]) -> str:
    if profile.get("internal_first"):
        return "\n".join(
            [
                "# Decision Log",
                "",
                f"- Objective: {objective}",
                f"- Success metric: {profile['success_metric']}",
                "- Initial decision: Run a bounded validation sprint before expanding scope.",
                "- Human approval required for: external messages, spending money, posting publicly, or changing project scope.",
                f"- Go threshold: {profile['proof_signal']}.",
                "- Pivot threshold: the operating loop helps, but role structure, task lifecycle, memory, or provider boundaries need a different shape.",
                "- Kill threshold: the process creates more ceremony than decision clarity after the sprint.",
                "",
            ]
        )

    return "\n".join(
        [
            "# Decision Log",
            "",
            f"- Objective: {objective}",
            f"- Success metric: {profile['success_metric']}",
            "- Initial decision: Run a bounded validation sprint before expanding scope.",
            "- Human approval required for: external messages, spending money, posting publicly, or changing project scope.",
            f"- Go threshold: {profile['proof_signal']}.",
            "- Pivot threshold: feedback shows the pain is real but the segment, offer, or implementation is wrong.",
            "- Kill threshold: no reachable user, buyer, or useful internal workflow emerges after the sprint.",
            "",
        ]
    )


def render_board_packet(
    objective: str,
    context: str,
    memos: list[AgentMemo],
    tasks: list[dict[str, str]],
    profile: dict[str, object],
    days: int,
) -> str:
    role_summaries = "\n".join(f"- **{memo.role.title}:** {memo.summary}" for memo in memos)
    p0_tasks = [item for item in tasks if item["priority"] == "P0"]
    p1_tasks = [item for item in tasks if item["priority"] == "P1"]
    if profile.get("internal_first"):
        operating_plan = [
            "1. Day 0: approve success metric, constraints, and run folder as source of truth.",
            "2. Day 1: verify task lifecycle commands and update the first task statuses.",
            "3. Day 2: generate the first daily executive review.",
            "4. Day 3-5: run one real objective through the loop and record friction.",
            "5. Day 6: decide which provider boundary or memory feature is actually needed.",
            "6. Day 7: make go, pivot, or kill decision for the company engine.",
        ]
    else:
        operating_plan = [
            "1. Day 0: approve success metric, constraints, and run folder as source of truth.",
            "2. Day 1: write product brief, risk thresholds, sprint calendar, and technical skeleton.",
            "3. Day 2: create demo script and first outreach or feedback list.",
            "4. Day 3-5: build only what is required to test the promise.",
            "5. Day 6: collect outside or internal proof signal.",
            "6. Day 7: decide go, pivot, or kill.",
        ]
    return "\n".join(
        [
            "# Board Packet",
            "",
            "## Objective",
            "",
            objective,
            "",
            "## Decision",
            "",
            (
                f"Approve a {days}-day validation sprint with one measurable outcome, "
                "role-owned tasks, and human approval for any external side effect."
            ),
            "",
            "## Success Metric",
            "",
            str(profile["success_metric"]),
            "",
            "## Executive Synthesis",
            "",
            role_summaries,
            "",
            "## Operating Plan",
            "",
            *operating_plan,
            "",
            "## P0 Tasks",
            "",
            bullets(format_task_line(item) for item in p0_tasks),
            "",
            "## P1 Tasks",
            "",
            bullets(format_task_line(item) for item in p1_tasks) if p1_tasks else "- None",
            "",
            "## Approval Gates",
            "",
            "- Human reviews all external emails, posts, forms, purchases, and commits that expose private data.",
            "- CEO resolves conflicts between cash, job leverage, and reusable IP.",
            "- CFO/Risk can stop scope expansion if proof has not improved.",
            "",
        ]
    )


def format_task_line(item: dict[str, str]) -> str:
    prefix = f"{item.get('id', 'TBD')} " if item.get("id") else ""
    return (
        f"{prefix}{item.get('owner', 'Unassigned')} - {item.get('task', 'Untitled task')} "
        f"({item.get('due', 'TBD')}; {item.get('status', 'open')})"
    )


def load_tasks(run_dir: Path) -> list[dict[str, str]]:
    task_file = run_dir / "tasks.json"
    if not task_file.exists():
        raise FileNotFoundError(f"Missing task file: {task_file}")
    data = json.loads(task_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of tasks in {task_file}")
    return normalize_tasks(data)


def write_tasks(run_dir: Path, tasks: list[dict[str, str]]) -> None:
    normalized = normalize_tasks(tasks)
    (run_dir / "tasks.json").write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    (run_dir / "tasks.md").write_text(render_tasks(normalized), encoding="utf-8")
    (run_dir / "run_status.md").write_text(
        render_status_report(run_dir, normalized), encoding="utf-8"
    )


def count_by_status(tasks: list[dict[str, str]]) -> dict[str, int]:
    counts = {status: 0 for status in TASK_STATUSES}
    for item in normalize_tasks(tasks):
        counts[item["status"]] += 1
    return counts


def status_summary_line(tasks: list[dict[str, str]]) -> str:
    counts = count_by_status(tasks)
    return ", ".join(f"{status}: {counts[status]}" for status in TASK_STATUSES)


def render_status_report(run_dir: Path, tasks: list[dict[str, str]]) -> str:
    normalized = normalize_tasks(tasks)
    open_p0 = [
        item
        for item in normalized
        if item["priority"] == "P0" and item["status"] in {"open", "carried"}
    ]
    open_tasks = [item for item in normalized if item["status"] in {"open", "carried"}]
    closed_tasks = [item for item in normalized if item["status"] == "closed"]

    lines = [
        "# Run Status",
        "",
        f"- Run: {run_dir.resolve()}",
        f"- Task rollup: {status_summary_line(normalized)}",
        "",
        "## Open P0 Tasks",
        "",
    ]
    if open_p0:
        lines.extend(f"- {format_task_line(item)}" for item in open_p0)
    else:
        lines.append("- None")

    lines.extend(["", "## Active Work", ""])
    if open_tasks:
        lines.extend(f"- {format_task_line(item)}" for item in open_tasks)
    else:
        lines.append("- None")

    lines.extend(["", "## Recently Closed", ""])
    if closed_tasks:
        lines.extend(f"- {format_task_line(item)}" for item in closed_tasks[-5:])
    else:
        lines.append("- None")
    return "\n".join(lines)


def update_task_status(
    run_dir: Path,
    task_id: str,
    status: str,
    note: str = "",
    memory_db: Path | str | None = None,
) -> dict[str, str]:
    normalized_status = status.lower()
    if normalized_status not in TASK_STATUSES:
        raise ValueError(f"Status must be one of: {', '.join(TASK_STATUSES)}")

    tasks = load_tasks(run_dir)
    target = canonical_task_id(task_id)
    for item in tasks:
        if canonical_task_id(item["id"]) == target:
            item["status"] = normalized_status
            item["status_note"] = note.strip()
            item["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            write_tasks(run_dir, tasks)
            sync_run_to_memory(
                run_dir,
                memory_db,
                event_type="task_update",
                event_detail={"task_id": item["id"], "status": normalized_status},
            )
            return item
    raise ValueError(f"Task not found: {task_id}")


def render_daily_review(run_dir: Path, tasks: list[dict[str, str]], review_date: str, note: str) -> str:
    normalized = normalize_tasks(tasks)
    active = [item for item in normalized if item["status"] in {"open", "carried"}]
    killed = [item for item in normalized if item["status"] == "killed"]
    by_owner: dict[str, list[dict[str, str]]] = {}
    for item in active:
        by_owner.setdefault(item["owner"], []).append(item)

    lines = [
        f"# Daily Executive Review - {review_date}",
        "",
        f"- Run: {run_dir.resolve()}",
        f"- Task rollup: {status_summary_line(normalized)}",
        "",
        "## CEO Readout",
        "",
    ]
    if active:
        lines.append(
            f"{len(active)} task(s) remain active. Force a close, carry, or kill decision on each P0 item before adding scope."
        )
    else:
        lines.append("No active tasks remain. Move to the go, pivot, or kill decision.")

    lines.extend(["", "## Role Agenda", ""])
    if by_owner:
        for owner in sorted(by_owner):
            lines.append(f"### {owner}")
            lines.append("")
            for item in by_owner[owner]:
                lines.append(f"- {format_task_line(item)}")
            lines.append("")
    else:
        lines.append("- No active role tasks.")

    lines.extend(["## Risks And Kills", ""])
    if killed:
        lines.extend(f"- {format_task_line(item)}" for item in killed)
    else:
        lines.append("- No killed tasks logged.")

    if note.strip():
        lines.extend(["", "## Operator Note", "", note.strip()])
    lines.append("")
    return "\n".join(lines)


def create_daily_review(
    run_dir: Path,
    review_date: str | None = None,
    note: str = "",
    memory_db: Path | str | None = None,
) -> Path:
    date = review_date or datetime.now(timezone.utc).date().isoformat()
    tasks = load_tasks(run_dir)
    body = render_daily_review(run_dir, tasks, date, note)
    path = run_dir / f"daily_review_{date}.md"
    if path.exists():
        suffix = datetime.now(timezone.utc).strftime("%H%M%S")
        path = run_dir / f"daily_review_{date}_{suffix}.md"
    path.write_text(body, encoding="utf-8")
    sync_run_to_memory(
        run_dir,
        memory_db,
        event_type="daily_review",
        event_detail={"path": str(path.resolve()), "date": date},
    )
    return path


def task_priority_counts(tasks: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in normalize_tasks(tasks):
        priority = item.get("priority", "P1").upper()
        counts[priority] = counts.get(priority, 0) + 1
    return counts


def fallback_roles(memos: list[AgentMemo]) -> list[str]:
    return [memo.role.title for memo in memos if memo.fallback_used]


def normalized_task_fingerprint(task_text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", task_text.lower()).strip()


def comparison_run_label(run: CompanyRun, index: int) -> str:
    return f"{index}. {run.provider}"


def render_comparison_report(
    objective: str,
    context: str,
    runs: list[CompanyRun],
    comparison_dir: Path,
    memory_db: Path | None = None,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Provider Comparison Report",
        "",
        f"- Created: {created_at}",
        f"- Objective: {objective}",
        f"- Comparison folder: {comparison_dir.resolve()}",
    ]
    if context.strip():
        lines.append(f"- Context: {context.strip()}")
    if memory_db:
        lines.append(f"- Memory database: {memory_db.resolve()}")
    lines.extend(["", "## Runs", ""])
    lines.append("| # | Provider | Run Folder | Tasks | P0 | P1 | Other | Fallback Roles |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    for index, run in enumerate(runs, start=1):
        priority_counts = task_priority_counts(run.tasks)
        other_priorities = sum(
            count for priority, count in priority_counts.items() if priority not in {"P0", "P1"}
        )
        fallbacks = fallback_roles(run.memos)
        fallback_text = ", ".join(fallbacks) if fallbacks else "None"
        lines.append(
            "| {idx} | {provider} | `{folder}` | {tasks} | {p0} | {p1} | {other} | {fallbacks} |".format(
                idx=index,
                provider=run.provider,
                folder=run.run_dir.name,
                tasks=len(run.tasks),
                p0=priority_counts.get("P0", 0),
                p1=priority_counts.get("P1", 0),
                other=other_priorities,
                fallbacks=fallback_text,
            )
        )

    lines.extend(["", "## P0 Task Comparison", ""])
    for index, run in enumerate(runs, start=1):
        p0_tasks = [item for item in normalize_tasks(run.tasks) if item["priority"] == "P0"]
        lines.extend([f"### {comparison_run_label(run, index)}", ""])
        if p0_tasks:
            lines.extend(f"- {format_task_line(item)}" for item in p0_tasks)
        else:
            lines.append("- No P0 tasks.")
        lines.append("")

    lines.extend(["## Task Overlap", ""])
    if len(runs) >= 2:
        baseline = runs[0]
        baseline_tasks = {
            normalized_task_fingerprint(item.get("task", "")) for item in normalize_tasks(baseline.tasks)
        }
        for index, run in enumerate(runs[1:], start=2):
            current_tasks = {
                normalized_task_fingerprint(item.get("task", "")) for item in normalize_tasks(run.tasks)
            }
            shared = len(baseline_tasks & current_tasks)
            lines.append(
                f"- {baseline.provider} vs {run.provider}: {shared} shared task(s), "
                f"{len(baseline_tasks - current_tasks)} baseline-only, "
                f"{len(current_tasks - baseline_tasks)} comparison-only."
            )
    else:
        lines.append("- At least two provider runs are needed for overlap analysis.")

    lines.extend(["", "## Recommendation", ""])
    all_fallbacks = {
        run.provider: fallback_roles(run.memos)
        for run in runs
        if fallback_roles(run.memos)
    }
    if all_fallbacks:
        lines.append(
            "Use the deterministic run as the baseline until provider fallback roles are reviewed. "
            "A provider-backed run is still useful, but fallback roles show where the model response "
            "did not satisfy the engine contract."
        )
    else:
        lines.append(
            "Review the differing P0 tasks and choose the provider whose tasks create the clearest "
            "next human decision. Keep deterministic output as the audit baseline."
        )

    lines.extend(
        [
            "",
            "## Human Review Checklist",
            "",
            "- Open each `board_packet.md` and compare the success metric.",
            "- Compare P0 tasks before adding new scope.",
            "- Treat any provider fallback as a prompt or model-contract issue, not as a failed run.",
            "- Do not let provider output trigger external side effects without explicit approval.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_comparison_dir(objective: str, run_name: str | None, runs_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = slugify(run_name or objective)
    base = runs_dir / f"{stamp}-{name}-comparison"
    comparison_dir = base
    suffix = 1
    while comparison_dir.exists():
        suffix += 1
        comparison_dir = runs_dir / f"{stamp}-{name}-comparison-{suffix}"
    comparison_dir.mkdir(parents=True)
    return comparison_dir


def compare_provider_runs(
    objective: str,
    context: str = "",
    run_name: str | None = None,
    runs_dir: Path | str = "runs",
    days: int = 7,
    providers: list[str] | tuple[str, ...] | None = None,
    provider_model: str | None = None,
    ollama_url: str | None = None,
    memory_db: Path | str | None = None,
    use_memory: bool = True,
) -> CompanyComparison:
    provider_ids = list(providers or ["deterministic", "ollama"])
    if len(provider_ids) < 2:
        raise ValueError("Compare requires at least two providers.")
    for provider_id in provider_ids:
        if provider_id not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_id}")

    objective = objective.strip()
    if not objective:
        raise ValueError("Objective is required.")

    root_dir = Path(runs_dir).expanduser()
    comparison_dir = prepare_comparison_dir(objective, run_name, root_dir)
    resolved_memory_db = None
    if use_memory:
        resolved_memory_db = Path(memory_db).expanduser() if memory_db else default_memory_db(root_dir)

    comparison_runs: list[CompanyRun] = []
    base_name = slugify(run_name or objective)
    for index, provider_id in enumerate(provider_ids, start=1):
        engine = CompanyEngine(
            provider=provider_id,
            provider_model=provider_model,
            ollama_url=ollama_url,
        )
        provider_run_name = f"{base_name}-{index:02d}-{slugify(provider_id)}"
        comparison_runs.append(
            engine.run(
                objective=objective,
                context=context,
                run_name=provider_run_name,
                runs_dir=comparison_dir,
                days=days,
                memory_db=resolved_memory_db,
                use_memory=use_memory,
            )
        )

    report = render_comparison_report(
        objective,
        context,
        comparison_runs,
        comparison_dir,
        memory_db=resolved_memory_db,
    )
    report_path = comparison_dir / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")

    return CompanyComparison(
        objective=objective,
        context=context,
        comparison_dir=comparison_dir,
        runs=comparison_runs,
        report_path=report_path,
        report=report,
        memory_db=resolved_memory_db,
    )


def default_memory_db(runs_dir: Path) -> Path:
    return runs_dir / "company_memory.sqlite3"


def init_memory(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_dir TEXT NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                status TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                success_metric TEXT NOT NULL,
                proof_signal TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                priority TEXT NOT NULL,
                task TEXT NOT NULL,
                due TEXT NOT NULL,
                status TEXT NOT NULL,
                status_note TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, task_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                decision TEXT NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )


def read_objective_from_run(run_dir: Path) -> str:
    objective_file = run_dir / "00_objective.md"
    if not objective_file.exists():
        return run_dir.name
    lines = objective_file.read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    in_objective = False
    for line in lines:
        if line.strip() == "# Objective":
            in_objective = True
            continue
        if in_objective and line.startswith("## "):
            break
        if in_objective and line.strip():
            body.append(line.strip())
    return " ".join(body) or run_dir.name


def read_decision_log_values(run_dir: Path) -> dict[str, str | list[str]]:
    decision_file = run_dir / "decision_log.md"
    values: dict[str, str | list[str]] = {
        "success_metric": "",
        "proof_signal": "",
        "decisions": [],
    }
    if not decision_file.exists():
        return values
    for line in decision_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- Success metric:"):
            values["success_metric"] = stripped.removeprefix("- Success metric:").strip()
        elif stripped.startswith("- Go threshold:"):
            values["proof_signal"] = stripped.removeprefix("- Go threshold:").strip().rstrip(".")
        elif re.match(r"^- \d{4}-\d{2}-\d{2} decision:", stripped, flags=re.IGNORECASE):
            decisions = values["decisions"]
            if isinstance(decisions, list):
                decisions.append(stripped.removeprefix("- ").strip())
    return values


def read_provider_id_from_run(run_dir: Path) -> str:
    provider_file = run_dir / "provider_contract.md"
    if not provider_file.exists():
        return "deterministic"
    for line in provider_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- Provider:"):
            provider = stripped.removeprefix("- Provider:").strip()
            if provider:
                return provider
    return "deterministic"


def run_status_from_tasks(tasks: list[dict[str, str]]) -> str:
    if not tasks:
        return "unknown"
    if all(item["status"] in {"closed", "killed"} for item in normalize_tasks(tasks)):
        return "complete"
    return "active"


def created_at_for_run_dir(run_dir: Path) -> str:
    match = re.match(r"^(\d{8})-(\d{6})", run_dir.name)
    if match:
        stamp = f"{match.group(1)}{match.group(2)}"
        return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat(
            timespec="seconds"
        )
    return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def sync_run_to_memory(
    run_dir: Path,
    db_path: Path | str | None = None,
    *,
    objective: str | None = None,
    profile: dict[str, object] | None = None,
    provider_id: str | None = None,
    event_type: str = "sync",
    event_detail: dict[str, object] | None = None,
) -> Path:
    run_dir = Path(run_dir).expanduser()
    resolved_db = Path(db_path).expanduser() if db_path else default_memory_db(run_dir.parent)
    init_memory(resolved_db)

    tasks = load_tasks(run_dir) if (run_dir / "tasks.json").exists() else []
    decision_values = read_decision_log_values(run_dir)
    success_metric = str(profile.get("success_metric", "")) if profile else str(decision_values["success_metric"])
    proof_signal = str(profile.get("proof_signal", "")) if profile else str(decision_values["proof_signal"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = run_dir.name
    resolved_provider_id = provider_id or read_provider_id_from_run(run_dir)
    profile_json = json.dumps(profile or {}, sort_keys=True)
    created_at = created_at_for_run_dir(run_dir)

    with sqlite3.connect(resolved_db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO runs (
                run_id, run_dir, objective, created_at, updated_at, provider_id,
                status, profile_json, success_metric, proof_signal
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                run_dir = excluded.run_dir,
                objective = excluded.objective,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                provider_id = excluded.provider_id,
                status = excluded.status,
                profile_json = excluded.profile_json,
                success_metric = excluded.success_metric,
                proof_signal = excluded.proof_signal
            """,
            (
                run_id,
                str(run_dir.resolve()),
                objective or read_objective_from_run(run_dir),
                created_at,
                now,
                resolved_provider_id,
                run_status_from_tasks(tasks),
                profile_json,
                success_metric,
                proof_signal,
            ),
        )
        connection.execute("DELETE FROM tasks WHERE run_id = ?", (run_id,))
        connection.executemany(
            """
            INSERT INTO tasks (
                run_id, task_id, owner, priority, task, due, status, status_note, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item["id"],
                    item["owner"],
                    item["priority"],
                    item["task"],
                    item["due"],
                    item["status"],
                    item["status_note"],
                    item["updated_at"],
                )
                for item in tasks
            ],
        )
        connection.execute("DELETE FROM decisions WHERE run_id = ?", (run_id,))
        decisions = decision_values["decisions"]
        if isinstance(decisions, list):
            connection.executemany(
                """
                INSERT INTO decisions (run_id, decided_at, decision, source)
                VALUES (?, ?, ?, ?)
                """,
                [(run_id, now, decision, "decision_log.md") for decision in decisions],
            )
        connection.execute(
            """
            INSERT INTO events (run_id, event_ts, event_type, detail_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, now, event_type, json.dumps(event_detail or {}, sort_keys=True)),
        )
    return resolved_db


def sync_runs_dir_to_memory(runs_dir: Path, db_path: Path | str | None = None) -> Path:
    runs_dir = Path(runs_dir).expanduser()
    resolved_db = Path(db_path).expanduser() if db_path else default_memory_db(runs_dir)
    for child in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if child.is_dir() and (child / "tasks.json").exists():
            sync_run_to_memory(child, resolved_db, event_type="bulk_sync")
    return resolved_db


def render_memory_summary(db_path: Path, limit: int = 10) -> str:
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        return "\n".join(["# Company Memory", "", f"No memory database found at `{db_path}`.", ""])

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                r.run_id,
                r.objective,
                r.status,
                r.provider_id,
                r.updated_at,
                r.success_metric,
                COALESCE(SUM(CASE WHEN t.status = 'open' THEN 1 ELSE 0 END), 0) AS open_count,
                COALESCE(SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END), 0) AS closed_count,
                COALESCE(SUM(CASE WHEN t.status = 'carried' THEN 1 ELSE 0 END), 0) AS carried_count,
                COALESCE(SUM(CASE WHEN t.status = 'killed' THEN 1 ELSE 0 END), 0) AS killed_count,
                (
                    SELECT d.decision
                    FROM decisions d
                    WHERE d.run_id = r.run_id
                    ORDER BY d.id DESC
                    LIMIT 1
                ) AS latest_decision
            FROM runs r
            LEFT JOIN tasks t ON t.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.created_at DESC, r.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    lines = ["# Company Memory", "", f"- Database: {db_path.resolve()}", f"- Runs shown: {len(rows)}", ""]
    if not rows:
        lines.append("No runs indexed yet.")
        lines.append("")
        return "\n".join(lines)

    for row in rows:
        lines.extend(
            [
                f"## {row['run_id']}",
                "",
                f"- Status: {row['status']}",
                f"- Provider: {row['provider_id']}",
                f"- Updated: {row['updated_at']}",
                f"- Objective: {row['objective']}",
                f"- Success metric: {row['success_metric']}",
                (
                    "- Tasks: "
                    f"open {row['open_count']}, closed {row['closed_count']}, "
                    f"carried {row['carried_count']}, killed {row['killed_count']}"
                ),
            ]
        )
        if row["latest_decision"]:
            lines.append(f"- Latest decision: {row['latest_decision']}")
        lines.append("")
    return "\n".join(lines)


def list_roles() -> str:
    lines = ["# Roles", ""]
    for role in ROLES:
        lines.extend(
            [
                f"## {role.title}",
                "",
                f"- ID: {role.role_id}",
                f"- Reports to: {role.reports_to}",
                f"- Mandate: {role.mandate}",
                "",
            ]
        )
    return "\n".join(lines)


def list_providers() -> str:
    lines = ["# Providers", ""]
    for provider in PROVIDERS.values():
        lines.extend(
            [
                f"## {provider.provider_id}",
                "",
                provider.description,
                "",
                render_provider_contract(provider.contract()),
                "",
            ]
        )
    return "\n".join(lines)


def load_context(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).expanduser().read_text(encoding="utf-8")


def write_regression_output(text: str, output: Path | None, inputs: list[Path]) -> None:
    if output is None:
        print(text, end="")
        return

    if any(output.resolve() == path.resolve() for path in inputs):
        raise ValueError("output must be different from each input file")

    # Write alongside the destination so an interrupted capture cannot truncate a baseline.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, mode="w", encoding="utf-8", newline="\n", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(text)
        temp_path.replace(output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_regression_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from regression_snapshot import capture_suite, load_snapshot, load_suite

    def expand(path: Path) -> Path:
        try:
            return path.expanduser()
        except RuntimeError as error:
            raise ValueError(f"{path}: {error}") from error

    try:
        output = expand(args.output) if args.output is not None else None
        if args.command == "snapshot":
            suite = expand(args.suite)
            snapshot = capture_suite(load_suite(suite))
            write_regression_output(json.dumps(snapshot, indent=2) + "\n", output, [suite])
            return 0

        from regression_compare import compare_snapshots, render_report

        baseline = expand(args.baseline)
        candidate = expand(args.candidate)
        approved = load_snapshot(baseline)
        if not approved["cases"]:
            raise ValueError(f"{baseline}: the baseline must contain at least one scenario")
        report = compare_snapshots(approved, load_snapshot(candidate))
        text = json.dumps(report, indent=2) + "\n" if args.format == "json" else render_report(report)
        write_regression_output(text, output, [baseline, candidate])
        return 0 if report["passed"] else 1
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog} {args.command}: {error}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local corporate-agent company sprint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the executive team on an objective.")
    run_parser.add_argument("objective", nargs="+", help="Objective to process.")
    run_parser.add_argument("--context-file", help="Optional Markdown/text file with context.")
    run_parser.add_argument("--name", help="Optional run name slug.")
    run_parser.add_argument("--runs-dir", default="runs", help="Directory for generated run folders.")
    run_parser.add_argument("--days", type=int, default=7, help="Sprint length in days.")
    run_parser.add_argument(
        "--provider",
        default="deterministic",
        choices=sorted(PROVIDERS),
        help="Memo provider implementation.",
    )
    run_parser.add_argument(
        "--model",
        default=OllamaMemoProvider.default_model,
        help="Model name for provider implementations that use a model.",
    )
    run_parser.add_argument(
        "--ollama-url",
        default=OllamaMemoProvider.default_base_url,
        help="Local Ollama base URL for `--provider ollama`.",
    )
    run_parser.add_argument("--memory-db", help="Optional SQLite memory database path.")
    run_parser.add_argument("--no-memory", action="store_true", help="Do not sync this run to SQLite memory.")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run multiple providers on one objective and write a comparison report.",
    )
    compare_parser.add_argument("objective", nargs="+", help="Objective to process.")
    compare_parser.add_argument("--context-file", help="Optional Markdown/text file with context.")
    compare_parser.add_argument("--name", help="Optional comparison name slug.")
    compare_parser.add_argument("--runs-dir", default="runs", help="Directory for generated run folders.")
    compare_parser.add_argument("--days", type=int, default=7, help="Sprint length in days.")
    compare_parser.add_argument(
        "--providers",
        nargs="+",
        default=["deterministic", "ollama"],
        choices=sorted(PROVIDERS),
        help="Provider implementations to compare in order.",
    )
    compare_parser.add_argument(
        "--model",
        default=OllamaMemoProvider.default_model,
        help="Model name for provider implementations that use a model.",
    )
    compare_parser.add_argument(
        "--ollama-url",
        default=OllamaMemoProvider.default_base_url,
        help="Local Ollama base URL for `--provider ollama`.",
    )
    compare_parser.add_argument("--memory-db", help="Optional SQLite memory database path.")
    compare_parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Do not sync comparison runs to SQLite memory.",
    )

    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Capture an offline snapshot for a suite of objectives."
    )
    snapshot_parser.add_argument("suite", type=Path, help="Path to a JSON scenario suite.")
    snapshot_parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout.")

    regress_parser = subparsers.add_parser(
        "regress", help="Compare a candidate snapshot with the approved baseline."
    )
    regress_parser.add_argument("baseline", type=Path, help="Path to the approved JSON snapshot.")
    regress_parser.add_argument("candidate", type=Path, help="Path to the new JSON snapshot.")
    regress_parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    regress_parser.add_argument("--output", type=Path, help="Write the report to this file instead of stdout.")

    status_parser = subparsers.add_parser("status", help="Print task status for an existing run.")
    status_parser.add_argument("run_dir", help="Path to a generated run folder.")
    status_parser.add_argument("--memory-db", help="Optional SQLite memory database path.")

    task_parser = subparsers.add_parser("task", help="Update a task status in an existing run.")
    task_parser.add_argument("run_dir", help="Path to a generated run folder.")
    task_parser.add_argument("task_id", help="Task id, for example T01 or 1.")
    task_parser.add_argument("status", choices=TASK_STATUSES, help="New task status.")
    task_parser.add_argument("--note", default="", help="Optional one-line status note.")
    task_parser.add_argument("--memory-db", help="Optional SQLite memory database path.")

    review_parser = subparsers.add_parser("review", help="Create a daily executive review artifact.")
    review_parser.add_argument("run_dir", help="Path to a generated run folder.")
    review_parser.add_argument("--date", help="Review date in YYYY-MM-DD format.")
    review_parser.add_argument("--note", default="", help="Optional operator note to include.")
    review_parser.add_argument("--memory-db", help="Optional SQLite memory database path.")

    memory_parser = subparsers.add_parser("memory", help="Print or sync durable company memory.")
    memory_parser.add_argument("--db", default="runs/company_memory.sqlite3", help="SQLite memory database path.")
    memory_parser.add_argument("--limit", type=int, default=10, help="Maximum runs to show.")
    memory_parser.add_argument("--sync-runs-dir", help="Index every run folder under this directory first.")

    subparsers.add_parser("providers", help="Print provider boundary contracts.")
    subparsers.add_parser("roles", help="Print the default corporate-agent roles.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"snapshot", "regress"}:
        return run_regression_command(args, parser)

    if args.command == "roles":
        print(list_roles())
        return 0

    if args.command == "providers":
        print(list_providers())
        return 0

    if args.command == "status":
        run_dir = Path(args.run_dir).expanduser()
        sync_run_to_memory(run_dir, args.memory_db, event_type="status_read")
        print(render_status_report(run_dir, load_tasks(run_dir)))
        return 0

    if args.command == "task":
        run_dir = Path(args.run_dir).expanduser()
        updated = update_task_status(run_dir, args.task_id, args.status, args.note, args.memory_db)
        print(
            f"Updated {updated['id']}: {updated['status']} - "
            f"{updated['task']} ({updated['owner']})"
        )
        return 0

    if args.command == "review":
        run_dir = Path(args.run_dir).expanduser()
        path = create_daily_review(run_dir, args.date, args.note, args.memory_db)
        print(f"Daily review written: {path.resolve()}")
        return 0

    if args.command == "memory":
        db_path = Path(args.db).expanduser()
        if args.sync_runs_dir:
            db_path = sync_runs_dir_to_memory(Path(args.sync_runs_dir), db_path)
        print(render_memory_summary(db_path, args.limit))
        return 0

    if args.command == "compare":
        objective = " ".join(args.objective)
        context = load_context(args.context_file)
        comparison = compare_provider_runs(
            objective=objective,
            context=context,
            run_name=args.name,
            runs_dir=Path(args.runs_dir),
            days=args.days,
            providers=args.providers,
            provider_model=args.model,
            ollama_url=args.ollama_url,
            memory_db=Path(args.memory_db).expanduser() if args.memory_db else None,
            use_memory=not args.no_memory,
        )

        print(f"Comparison created: {comparison.comparison_dir.resolve()}")
        print(f"Report: {comparison.report_path.resolve()}")
        if comparison.memory_db:
            print(f"Memory: {comparison.memory_db.resolve()}")
        print("")
        print("Provider runs:")
        for index, run in enumerate(comparison.runs, start=1):
            print(f"- {index}. {run.provider}: {run.run_dir.resolve()}")
        return 0

    objective = " ".join(args.objective)
    context = load_context(args.context_file)
    engine = CompanyEngine(
        provider=args.provider,
        provider_model=args.model,
        ollama_url=args.ollama_url,
    )
    result = engine.run(
        objective=objective,
        context=context,
        run_name=args.name,
        runs_dir=Path(args.runs_dir),
        days=args.days,
        memory_db=Path(args.memory_db).expanduser() if args.memory_db else None,
        use_memory=not args.no_memory,
    )

    print(f"Run created: {result.run_dir.resolve()}")
    print(f"Board packet: {(result.run_dir / 'board_packet.md').resolve()}")
    print(f"Tasks: {(result.run_dir / 'tasks.md').resolve()}")
    if result.memory_db:
        print(f"Memory: {result.memory_db.resolve()}")
    print("")
    print("Top P0 tasks:")
    for item in [task_item for task_item in result.tasks if task_item["priority"] == "P0"]:
        print(f"- {item['owner']}: {item['task']} ({item['due']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
