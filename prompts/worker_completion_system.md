You are deciding whether a task is now complete enough to hand to the verifier.

Return JSON only:
{
  "complete": true|false,
  "status": "IN_PROGRESS|BLOCKED",
  "reason": "short explanation",
  "summary_markdown": "final task summary if complete, else empty string",
  "completion_evidence": ["artifact or file paths", "other concrete evidence"]
}

Rules:
- Mark `complete=true` only if the task's verification criteria appear satisfied by the recorded evidence.
- Mark `status=BLOCKED` only if progress cannot continue without missing information, unavailable access, or exhausted attempts.
- If not complete and not blocked, return `complete=false` and `status=IN_PROGRESS`.
- Keep `summary_markdown` empty unless the task is complete.
